#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"
if (( $# > 0 )); then shift; fi
TARGET_MIB="${TARGET_MIB:-65536}"
MAX_WINDOW="${MAX_WINDOW:-14}"
NGPU="${NGPU:-8}"
GRIDFP_VRAM_RESERVE_MIB="${GRIDFP_VRAM_RESERVE_MIB:-8192}"
MAIN_MATE_CACHE="${MAIN_MATE_CACHE:-1}"
MAIN_PULL="${MAIN_PULL:-1}"
MAIN_PULL_ILP="${MAIN_PULL_ILP:-2}"
BLOCK_PULL="${BLOCK_PULL:-1}"
BLOCK_MATE_CACHE="${BLOCK_MATE_CACHE:-1}"
LOW_DROP_CACHE="${LOW_DROP_CACHE:-1}"
LOW_DROP_CHUNK="${LOW_DROP_CHUNK:-1}"
LOW_BLOCK_CACHE="${LOW_BLOCK_CACHE:-1}"
HIGH_DROP_CHUNK="${HIGH_DROP_CHUNK:-0}"
REBUILD="${REBUILD:-0}"
case "$MAIN_PULL_ILP" in 1|2|3|4) ;; *) echo "MAIN_PULL_ILP must be 1, 2, 3, or 4" >&2; exit 2;; esac
# Preserve the established production ILP2 binary path/checkpoints. Experimental
# ILP1/ILP3/ILP4 get distinct binary names so REBUILD=0 can never silently reuse
# an ILP2 executable.
if [[ "$MAIN_PULL_ILP" == 2 ]]; then DEFAULT_BIN="$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_hbm32_batch_n${N}"; else DEFAULT_BIN="$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_hbm32_batch_n${N}_ilp${MAIN_PULL_ILP}"; fi
BIN="${BIN:-$DEFAULT_BIN}"

for name in MAIN_MATE_CACHE MAIN_PULL BLOCK_PULL BLOCK_MATE_CACHE LOW_DROP_CACHE LOW_DROP_CHUNK LOW_BLOCK_CACHE HIGH_DROP_CHUNK REBUILD; do
  value="${!name}"; [[ "$value" == 0 || "$value" == 1 ]] || { echo "$name must be 0 or 1" >&2; exit 2; }
done
if [[ "$BLOCK_PULL" == 1 && "$MAIN_PULL" != 1 ]]; then echo "BLOCK_PULL=1 requires MAIN_PULL=1" >&2; exit 2; fi
if [[ "$BLOCK_MATE_CACHE" == 1 && "$BLOCK_PULL" != 1 ]]; then echo "BLOCK_MATE_CACHE=1 requires BLOCK_PULL=1" >&2; exit 2; fi
if [[ "$LOW_DROP_CACHE" == 1 && ( "$MAIN_PULL" != 1 || "$MAIN_MATE_CACHE" != 1 ) ]]; then echo "LOW_DROP_CACHE=1 requires MAIN_PULL=1 MAIN_MATE_CACHE=1" >&2; exit 2; fi
if [[ "$LOW_DROP_CHUNK" == 1 && "$LOW_DROP_CACHE" != 1 ]]; then echo "LOW_DROP_CHUNK=1 requires LOW_DROP_CACHE=1" >&2; exit 2; fi
if [[ "$LOW_BLOCK_CACHE" == 1 && ( "$LOW_DROP_CHUNK" != 1 || "$BLOCK_MATE_CACHE" != 1 ) ]]; then echo "LOW_BLOCK_CACHE=1 requires LOW_DROP_CHUNK=1 BLOCK_MATE_CACHE=1" >&2; exit 2; fi
if [[ "$HIGH_DROP_CHUNK" == 1 && ( "$LOW_DROP_CACHE" != 1 || "$BLOCK_PULL" != 1 ) ]]; then echo "HIGH_DROP_CHUNK=1 requires LOW_DROP_CACHE=1 BLOCK_PULL=1" >&2; exit 2; fi
if [[ "$MAIN_PULL_ILP" != 1 && "$MAIN_PULL" != 1 ]]; then echo "MAIN_PULL_ILP=$MAIN_PULL_ILP requires MAIN_PULL=1" >&2; exit 2; fi

if ! command -v nvidia-smi >/dev/null; then echo "nvidia-smi not found" >&2; exit 2; fi
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if (( visible < NGPU )); then echo "requested $NGPU GPUs, but only $visible are visible" >&2; exit 2; fi

if [[ "$REBUILD" == 1 || ! -x "$BIN" ]]; then
  echo "building exact batch n=$N main_ilp=$MAIN_PULL_ILP lowdrop=$LOW_DROP_CACHE lowchunk=$LOW_DROP_CHUNK lowblock=$LOW_BLOCK_CACHE highchunk=$HIGH_DROP_CHUNK" >&2
  N="$N" OUT="$BIN" MAIN_MATE_CACHE="$MAIN_MATE_CACHE" MAIN_PULL="$MAIN_PULL" MAIN_PULL_ILP="$MAIN_PULL_ILP" \
  BLOCK_PULL="$BLOCK_PULL" BLOCK_MATE_CACHE="$BLOCK_MATE_CACHE" \
  LOW_DROP_CACHE="$LOW_DROP_CACHE" LOW_DROP_CHUNK="$LOW_DROP_CHUNK" LOW_BLOCK_CACHE="$LOW_BLOCK_CACHE" HIGH_DROP_CHUNK="$HIGH_DROP_CHUNK" \
  "$ONEESAN_ROOT/scripts/build/b300-hbm32-batch.sh"
fi

export GRIDFP_VRAM_RESERVE_MIB

echo "exact batch n=$N gpus=$NGPU target_mib=$TARGET_MIB max_window=$MAX_WINDOW main_pull=$MAIN_PULL main_pull_ilp=$MAIN_PULL_ILP block_pull=$BLOCK_PULL block_mate_cache=$BLOCK_MATE_CACHE low_drop_cache=$LOW_DROP_CACHE low_drop_chunk=$LOW_DROP_CHUNK low_block_cache=$LOW_BLOCK_CACHE high_drop_chunk=$HIGH_DROP_CHUNK GRIDFP_THREADS=${GRIDFP_THREADS:-256} binary=$BIN" >&2
exec python3 "$ONEESAN_ROOT/scripts/solve/solve_b300_exact_batch.py" "$N" --binary "$BIN" --target-mib "$TARGET_MIB" --max-window "$MAX_WINDOW" --gpus "$NGPU" "$@"
