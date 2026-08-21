#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"
if (( $# > 0 )); then shift; fi
TARGET_MIB="${TARGET_MIB:-16384}"
MAX_WINDOW="${MAX_WINDOW:-14}"
NGPU="${NGPU:-8}"
GRIDFP_VRAM_RESERVE_MIB="${GRIDFP_VRAM_RESERVE_MIB:-8192}"
BIN="${BIN:-$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_hbm32_batch_n${N}}"

if ! command -v nvidia-smi >/dev/null; then
  echo "nvidia-smi not found" >&2
  exit 2
fi
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if (( visible < NGPU )); then
  echo "requested $NGPU GPUs, but only $visible are visible" >&2
  exit 2
fi

if [[ ! -x "$BIN" ]]; then
  echo "$BIN not found; building optimized batch binary for n=$N" >&2
  N="$N" OUT="$BIN" "$ONEESAN_ROOT/scripts/build/b300-hbm32-batch.sh"
fi

export GRIDFP_VRAM_RESERVE_MIB

exec python3 "$ONEESAN_ROOT/scripts/solve/solve_b300_exact_batch.py" "$N" \
  --binary "$BIN" \
  --target-mib "$TARGET_MIB" \
  --max-window "$MAX_WINDOW" \
  --gpus "$NGPU" \
  "$@"
