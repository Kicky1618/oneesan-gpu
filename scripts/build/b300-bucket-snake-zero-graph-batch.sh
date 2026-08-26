#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
N="${N:-27}"; W=$((N + 1)); ARCH="${ARCH:-native}"
LOW_LUT_K="${LOW_LUT_K:-$((W / 2))}"; HIGH_LUT_K="${HIGH_LUT_K:-$((W - LOW_LUT_K - 1))}"
TRANSPOSE_MODE="${TRANSPOSE_MODE:-pipeline}"; PM_ACCUM="${PM_ACCUM:-0}"; ZERO_HIGH_PLAN="${ZERO_HIGH_PLAN:-thread}"
if (( LOW_LUT_K <= 0 || HIGH_LUT_K <= 0 || LOW_LUT_K + HIGH_LUT_K + 1 != W )); then echo "invalid factor split" >&2; exit 2; fi
if [[ "$PM_ACCUM" != 0 && "$PM_ACCUM" != 1 ]]; then echo "PM_ACCUM must be 0 or 1" >&2; exit 2; fi
case "$ZERO_HIGH_PLAN" in thread) STEM="oneesan_cuda_gridfp_b300_bucket_snake_onepass_zero_graph_batch";; shared) STEM="oneesan_cuda_gridfp_b300_bucket_snake_onepass_zero_shared_graph_batch";; *) echo "ZERO_HIGH_PLAN must be thread or shared" >&2; exit 2;; esac
case "$TRANSPOSE_MODE" in
  sync) SRC_REL="src/cuda/b300/${STEM}.cu" ;;
  events) SRC_REL="src/cuda/b300/${STEM}_events.cu" ;;
  pipeline) SRC_REL="src/cuda/b300/${STEM}_pipeline.cu" ;;
  *) echo "TRANSPOSE_MODE must be sync, events, or pipeline" >&2; exit 2 ;;
esac
SUFFIX="_${ZERO_HIGH_PLAN}_${TRANSPOSE_MODE}"; if [[ "$PM_ACCUM" == 1 ]]; then SUFFIX="${SUFFIX}_pm"; fi
SRC="$(repo_path "$SRC_REL")"; OUT="$(build_path "${OUT:-oneesan_cuda_gridfp_b300_bucket_snake_onepass_zero_graph_batch${SUFFIX}_n${N}}")"
TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" \
  -DTARGET_W="$W" -DLOW_LUT_K="$LOW_LUT_K" -DHIGH_LUT_K="$HIGH_LUT_K" -DGPU_DIRECT_PM_ACCUM="$PM_ACCUM" \
  "$SRC" -o "$OUT"
echo "built $OUT (forward=orbit reverse=split54 closure=zero high_plan=$ZERO_HIGH_PLAN window=graph transpose=$TRANSPOSE_MODE pm_accum=$PM_ACCUM)"
