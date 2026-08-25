#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-27}"
W=$((N + 1))
ARCH="${ARCH:-native}"
LOW_LUT_K="${LOW_LUT_K:-$((W / 2))}"
HIGH_LUT_K="${HIGH_LUT_K:-$((W - LOW_LUT_K - 1))}"
TRANSPOSE_MODE="${TRANSPOSE_MODE:-events}"
ORBIT_CLOSURE_FUSE="${ORBIT_CLOSURE_FUSE:-0}"
WINDOW_MODE="${WINDOW_MODE:-direct}"
PM_ACCUM="${PM_ACCUM:-0}"

if (( LOW_LUT_K <= 0 || HIGH_LUT_K <= 0 || LOW_LUT_K + HIGH_LUT_K + 1 != W )); then
  echo "invalid factor split" >&2
  exit 2
fi
if [[ "$PM_ACCUM" != 0 && "$PM_ACCUM" != 1 ]]; then
  echo "PM_ACCUM must be 0 or 1" >&2
  exit 2
fi
if [[ "$ORBIT_CLOSURE_FUSE" != 0 && "$ORBIT_CLOSURE_FUSE" != 1 ]]; then
  echo "ORBIT_CLOSURE_FUSE must be 0 or 1" >&2
  exit 2
fi
if [[ "$WINDOW_MODE" != direct && "$WINDOW_MODE" != graph ]]; then
  echo "WINDOW_MODE must be direct or graph" >&2
  exit 2
fi
if [[ "$ORBIT_CLOSURE_FUSE" == 1 ]]; then
  if [[ "$WINDOW_MODE" == graph ]]; then
    case "$TRANSPOSE_MODE" in
      sync) SRC_REL="src/cuda/b300/oneesan_cuda_gridfp_b300_bucket_snake_onepass_graph_batch.cu" ;;
      events) SRC_REL="src/cuda/b300/oneesan_cuda_gridfp_b300_bucket_snake_onepass_graph_batch_events.cu" ;;
      pipeline) SRC_REL="src/cuda/b300/oneesan_cuda_gridfp_b300_bucket_snake_onepass_graph_batch_pipeline.cu" ;;
      *) echo "TRANSPOSE_MODE must be sync, events, or pipeline" >&2; exit 2 ;;
    esac
    STEM="oneesan_cuda_gridfp_b300_bucket_snake_onepass_graph_batch"
  else
    case "$TRANSPOSE_MODE" in
      sync) SRC_REL="src/cuda/b300/oneesan_cuda_gridfp_b300_bucket_snake_onepass_batch.cu" ;;
      events) SRC_REL="src/cuda/b300/oneesan_cuda_gridfp_b300_bucket_snake_onepass_batch_events.cu" ;;
      pipeline) SRC_REL="src/cuda/b300/oneesan_cuda_gridfp_b300_bucket_snake_onepass_batch_pipeline.cu" ;;
      *) echo "TRANSPOSE_MODE must be sync, events, or pipeline" >&2; exit 2 ;;
    esac
    STEM="oneesan_cuda_gridfp_b300_bucket_snake_onepass_batch"
  fi
  SUFFIX="_${TRANSPOSE_MODE}"
else
  if [[ "$WINDOW_MODE" != direct ]]; then
    echo "WINDOW_MODE=graph requires ORBIT_CLOSURE_FUSE=1" >&2
    exit 2
  fi
  case "$TRANSPOSE_MODE" in
    sync) SRC_REL="src/cuda/b300/oneesan_cuda_gridfp_b300_bucket_snake_fused_batch.cu" ;;
    events) SRC_REL="src/cuda/b300/oneesan_cuda_gridfp_b300_bucket_snake_fused_batch_events.cu" ;;
    pipeline) SRC_REL="src/cuda/b300/oneesan_cuda_gridfp_b300_bucket_snake_fused_batch_pipeline.cu" ;;
    *) echo "TRANSPOSE_MODE must be sync, events, or pipeline" >&2; exit 2 ;;
  esac
  SUFFIX="_${TRANSPOSE_MODE}"
  STEM="oneesan_cuda_gridfp_b300_bucket_snake_fused_batch"
fi
if [[ "$PM_ACCUM" == 1 ]]; then SUFFIX="${SUFFIX}_pm"; fi
SRC="$(repo_path "$SRC_REL")"
OUT="$(build_path "${OUT:-${STEM}${SUFFIX}_n${N}}")"

TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" \
  -DTARGET_W="$W" -DLOW_LUT_K="$LOW_LUT_K" -DHIGH_LUT_K="$HIGH_LUT_K" \
  -DGPU_DIRECT_PM_ACCUM="$PM_ACCUM" \
  "$SRC" -o "$OUT"
echo "built $OUT (reverse=fused transpose=$TRANSPOSE_MODE orbit_closure_fuse=$ORBIT_CLOSURE_FUSE window=$WINDOW_MODE pm_accum=$PM_ACCUM)"
