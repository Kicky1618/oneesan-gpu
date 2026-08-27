#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
N="${N:-27}"; W=$((N + 1)); ARCH="${ARCH:-native}"
LOW_LUT_K="${LOW_LUT_K:-$((W / 2))}"; HIGH_LUT_K="${HIGH_LUT_K:-$((W - LOW_LUT_K - 1))}"
TRANSPOSE_MODE="${TRANSPOSE_MODE:-pipeline}"; PM_ACCUM="${PM_ACCUM:-0}"
if (( LOW_LUT_K <= 0 || HIGH_LUT_K <= 0 || LOW_LUT_K + HIGH_LUT_K + 1 != W )); then echo "invalid factor split" >&2; exit 2; fi
if (( LOW_LUT_K > 14 || HIGH_LUT_K > 14 )); then echo "pattern10 depth8 requires half widths <=14" >&2; exit 2; fi
if [[ "$PM_ACCUM" != 0 && "$PM_ACCUM" != 1 ]]; then echo "PM_ACCUM must be 0 or 1" >&2; exit 2; fi
case "$TRANSPOSE_MODE" in
  sync) SRC_NAME="oneesan_cuda_gridfp_b300_bucket_snake_onepass_pattern10_depth8_graph_batch.cu" ;;
  events) SRC_NAME="oneesan_cuda_gridfp_b300_bucket_snake_onepass_pattern10_depth8_graph_batch_events.cu" ;;
  pipeline) SRC_NAME="oneesan_cuda_gridfp_b300_bucket_snake_onepass_pattern10_depth8_graph_batch_pipeline.cu" ;;
  *) echo "TRANSPOSE_MODE must be sync, events, or pipeline" >&2; exit 2 ;;
esac
SRC="$(repo_path "src/cuda/b300/$SRC_NAME")"; SUFFIX="_${TRANSPOSE_MODE}"; if [[ "$PM_ACCUM" == 1 ]]; then SUFFIX="${SUFFIX}_pm"; fi
OUT="$(build_path "${OUT:-oneesan_cuda_gridfp_b300_bucket_snake_onepass_pattern10_depth8_graph_batch${SUFFIX}_n${N}}")"
TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" \
  -DTARGET_W="$W" -DLOW_LUT_K="$LOW_LUT_K" -DHIGH_LUT_K="$HIGH_LUT_K" -DGPU_DIRECT_PM_ACCUM="$PM_ACCUM" \
  "$SRC" -o "$OUT"
echo "built $OUT (closure=pattern10+depth8 ordinary_scan=0 cross_scan=0 window=graph transpose=$TRANSPOSE_MODE pm_accum=$PM_ACCUM)"
