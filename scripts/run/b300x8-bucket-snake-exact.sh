#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"
if (( $# > 0 )); then shift; fi
NGPU="${NGPU:-8}"
TARGET_MIB="${TARGET_MIB:-16384}"
MAX_WINDOW="${MAX_WINDOW:-14}"
TRANSPOSE_MODE="${TRANSPOSE_MODE:-events}"
ORBIT_CLOSURE_FUSE="${ORBIT_CLOSURE_FUSE:-0}"
WINDOW_MODE="${WINDOW_MODE:-direct}"
PM_ACCUM="${PM_ACCUM:-0}"
BUCKET_TRANSPOSE_CHUNK_MIB="${BUCKET_TRANSPOSE_CHUNK_MIB:-1024}"
BUCKET_RESERVE_MIB="${BUCKET_RESERVE_MIB:-8192}"
if [[ "$ORBIT_CLOSURE_FUSE" != 0 && "$ORBIT_CLOSURE_FUSE" != 1 ]]; then
  echo "ORBIT_CLOSURE_FUSE must be 0 or 1" >&2
  exit 2
fi
if [[ "$WINDOW_MODE" != direct && "$WINDOW_MODE" != graph ]]; then
  echo "WINDOW_MODE must be direct or graph" >&2
  exit 2
fi
if [[ "$ORBIT_CLOSURE_FUSE" == 1 ]]; then
  SUFFIX="_${TRANSPOSE_MODE}"
  if [[ "$WINDOW_MODE" == graph ]]; then
    STEM="oneesan_cuda_gridfp_b300_bucket_snake_onepass_graph_batch"
    WORK_TAG="onepass_graph_${TRANSPOSE_MODE}"
  else
    STEM="oneesan_cuda_gridfp_b300_bucket_snake_onepass_batch"
    WORK_TAG="onepass_${TRANSPOSE_MODE}"
  fi
else
  if [[ "$WINDOW_MODE" != direct ]]; then
    echo "WINDOW_MODE=graph requires ORBIT_CLOSURE_FUSE=1" >&2
    exit 2
  fi
  SUFFIX="_${TRANSPOSE_MODE}"
  STEM="oneesan_cuda_gridfp_b300_bucket_snake_fused_batch"
  WORK_TAG="fused_${TRANSPOSE_MODE}"
fi
if [[ "$PM_ACCUM" == 1 ]]; then SUFFIX="${SUFFIX}_pm"; WORK_TAG="${WORK_TAG}_pm"; fi
BIN="${BIN:-$ONEESAN_BUILD_DIR/${STEM}${SUFFIX}_n${N}}"
WORK_DIR="${WORK_DIR:-$ONEESAN_ROOT/work/b300_bucket_snake_${WORK_TAG}_exact_n${N}}"

if (( NGPU != 8 )); then
  echo "snake bucket backend currently requires NGPU=8" >&2
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
  echo "$BIN not found; building snake batch binary for n=$N transpose=$TRANSPOSE_MODE onepass=$ORBIT_CLOSURE_FUSE window=$WINDOW_MODE pm=$PM_ACCUM" >&2
  N="$N" OUT="$BIN" TRANSPOSE_MODE="$TRANSPOSE_MODE" ORBIT_CLOSURE_FUSE="$ORBIT_CLOSURE_FUSE" WINDOW_MODE="$WINDOW_MODE" PM_ACCUM="$PM_ACCUM" \
    bash "$ONEESAN_ROOT/scripts/build/b300-bucket-snake-batch.sh"
fi

export BUCKET_TRANSPOSE_CHUNK_MIB BUCKET_RESERVE_MIB

exec python3 "$ONEESAN_ROOT/scripts/solve/solve_b300_exact_batch.py" "$N" \
  --binary "$BIN" \
  --work-dir "$WORK_DIR" \
  --target-mib "$TARGET_MIB" \
  --max-window "$MAX_WINDOW" \
  --gpus "$NGPU" \
  "$@"
