#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"
if (( $# > 0 )); then shift; fi
NGPU="${NGPU:-8}"
TARGET_MIB="${TARGET_MIB:-16384}"   # legacy batch-runner argument; bucket backend ignores it
MAX_WINDOW="${MAX_WINDOW:-14}"      # legacy batch-runner argument; bucket backend ignores it
TRANSPOSE_MODE="${TRANSPOSE_MODE:-events}"
BUCKET_TRANSPOSE_CHUNK_MIB="${BUCKET_TRANSPOSE_CHUNK_MIB:-1024}"
BUCKET_RESERVE_MIB="${BUCKET_RESERVE_MIB:-8192}"
BIN="${BIN:-$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_bucket_fused_batch_n${N}}"
WORK_DIR="${WORK_DIR:-$ONEESAN_ROOT/work/b300_bucket_exact_n${N}}"

if (( NGPU != 8 )); then
  echo "bucket backend currently requires NGPU=8" >&2
  exit 2
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

if [[ ! -x "$BIN" ]]; then
  echo "$BIN not found; building bucket-fused batch binary for n=$N transpose=$TRANSPOSE_MODE" >&2
  N="$N" OUT="$BIN" TRANSPOSE_MODE="$TRANSPOSE_MODE" \
    "$ONEESAN_ROOT/scripts/build/b300-bucket-fused-batch.sh"
fi

export BUCKET_TRANSPOSE_CHUNK_MIB BUCKET_RESERVE_MIB

exec python3 "$ONEESAN_ROOT/scripts/solve/solve_b300_exact_batch.py" "$N" \
  --binary "$BIN" \
  --work-dir "$WORK_DIR" \
  --target-mib "$TARGET_MIB" \
  --max-window "$MAX_WINDOW" \
  --gpus "$NGPU" \
  "$@"
