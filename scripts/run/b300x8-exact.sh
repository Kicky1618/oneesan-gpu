#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"
if (( $# > 0 )); then shift; fi
TARGET_MIB="${TARGET_MIB:-65536}"
# W28 production is compiled as 27..15 / 14..1. Keep the runtime cap aligned
# with that split rather than suggesting a wider, rank-unfriendly window.
MAX_WINDOW="${MAX_WINDOW:-14}"
NGPU="${NGPU:-8}"
GRIDFP_VRAM_RESERVE_MIB="${GRIDFP_VRAM_RESERVE_MIB:-8192}"
MAIN_MATE_CACHE="${MAIN_MATE_CACHE:-1}"
MAIN_PULL="${MAIN_PULL:-1}"
BLOCK_PULL="${BLOCK_PULL:-1}"
BLOCK_MATE_CACHE="${BLOCK_MATE_CACHE:-1}"
LOW_DROP_CACHE="${LOW_DROP_CACHE:-1}"
REBUILD="${REBUILD:-0}"
BIN="${BIN:-$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_hbm32_batch_n${N}}"

for name in MAIN_MATE_CACHE MAIN_PULL BLOCK_PULL BLOCK_MATE_CACHE LOW_DROP_CACHE REBUILD; do
  value="${!name}"
  [[ "$value" == 0 || "$value" == 1 ]] || { echo "$name must be 0 or 1" >&2; exit 2; }
done
if [[ "$BLOCK_PULL" == 1 && "$MAIN_PULL" != 1 ]]; then
  echo "BLOCK_PULL=1 requires MAIN_PULL=1" >&2; exit 2
fi
if [[ "$BLOCK_MATE_CACHE" == 1 && "$BLOCK_PULL" != 1 ]]; then
  echo "BLOCK_MATE_CACHE=1 requires BLOCK_PULL=1" >&2; exit 2
fi
if [[ "$LOW_DROP_CACHE" == 1 && ( "$MAIN_PULL" != 1 || "$MAIN_MATE_CACHE" != 1 ) ]]; then
  echo "LOW_DROP_CACHE=1 requires MAIN_PULL=1 MAIN_MATE_CACHE=1" >&2; exit 2
fi

if ! command -v nvidia-smi >/dev/null; then
  echo "nvidia-smi not found" >&2
  exit 2
fi
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if (( visible < NGPU )); then
  echo "requested $NGPU GPUs, but only $visible are visible" >&2
  exit 2
fi

if [[ "$REBUILD" == 1 || ! -x "$BIN" ]]; then
  echo "building optimized exact batch n=$N mainpull=$MAIN_PULL blockpull=$BLOCK_PULL blockmate=$BLOCK_MATE_CACHE lowdrop=$LOW_DROP_CACHE" >&2
  N="$N" OUT="$BIN" MAIN_MATE_CACHE="$MAIN_MATE_CACHE" \
  MAIN_PULL="$MAIN_PULL" BLOCK_PULL="$BLOCK_PULL" BLOCK_MATE_CACHE="$BLOCK_MATE_CACHE" \
  LOW_DROP_CACHE="$LOW_DROP_CACHE" \
  "$ONEESAN_ROOT/scripts/build/b300-hbm32-batch.sh"
fi

export GRIDFP_VRAM_RESERVE_MIB

echo "exact batch n=$N gpus=$NGPU target_mib=$TARGET_MIB max_window=$MAX_WINDOW main_mate_cache=$MAIN_MATE_CACHE main_pull=$MAIN_PULL block_pull=$BLOCK_PULL block_mate_cache=$BLOCK_MATE_CACHE low_drop_cache=$LOW_DROP_CACHE GRIDFP_THREADS=${GRIDFP_THREADS:-256} binary=$BIN" >&2
exec python3 "$ONEESAN_ROOT/scripts/solve/solve_b300_exact_batch.py" "$N" \
  --binary "$BIN" \
  --target-mib "$TARGET_MIB" \
  --max-window "$MAX_WINDOW" \
  --gpus "$NGPU" \
  "$@"
