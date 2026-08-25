#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-27}"
W=$((N + 1))
ARCH="${ARCH:-native}"
LOW_LUT_K="${LOW_LUT_K:-$((W / 2))}"
HIGH_LUT_K="${HIGH_LUT_K:-$((W - LOW_LUT_K - 1))}"
TRANSPOSE_MODE="${TRANSPOSE_MODE:-events}"
REVERSE_MODE="${REVERSE_MODE:-fused}"
PM_ACCUM="${PM_ACCUM:-0}"

if (( LOW_LUT_K <= 0 || HIGH_LUT_K <= 0 || LOW_LUT_K + HIGH_LUT_K + 1 != W )); then
  echo "invalid factor split" >&2
  exit 2
fi
if [[ "$PM_ACCUM" != 0 && "$PM_ACCUM" != 1 ]]; then
  echo "PM_ACCUM must be 0 or 1" >&2
  exit 2
fi
if [[ "$REVERSE_MODE" != atomic && "$REVERSE_MODE" != fused ]]; then
  echo "REVERSE_MODE must be atomic or fused" >&2
  exit 2
fi
case "$REVERSE_MODE/$TRANSPOSE_MODE" in
  atomic/sync) SRC_REL="src/cuda/b300/oneesan_cuda_gridfp_b300_bucket_snake_atomic.cu" ;;
  atomic/events) SRC_REL="src/cuda/b300/oneesan_cuda_gridfp_b300_bucket_snake_atomic_events.cu" ;;
  fused/sync) SRC_REL="src/cuda/b300/oneesan_cuda_gridfp_b300_bucket_snake_fused.cu" ;;
  fused/events) SRC_REL="src/cuda/b300/oneesan_cuda_gridfp_b300_bucket_snake_fused_events.cu" ;;
  *) echo "TRANSPOSE_MODE must be sync or events" >&2; exit 2 ;;
esac
SUFFIX="_${REVERSE_MODE}_${TRANSPOSE_MODE}"; if [[ "$PM_ACCUM" == 1 ]]; then SUFFIX="${SUFFIX}_pm"; fi
SRC="$(repo_path "$SRC_REL")"
OUT="$(build_path "${OUT:-oneesan_cuda_gridfp_b300_bucket_snake${SUFFIX}_n${N}}")"

TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" \
  -DTARGET_W="$W" -DLOW_LUT_K="$LOW_LUT_K" -DHIGH_LUT_K="$HIGH_LUT_K" \
  -DGPU_DIRECT_PM_ACCUM="$PM_ACCUM" \
  "$SRC" -o "$OUT"
echo "built $OUT (reverse=$REVERSE_MODE transpose=$TRANSPOSE_MODE pm_accum=$PM_ACCUM)"
