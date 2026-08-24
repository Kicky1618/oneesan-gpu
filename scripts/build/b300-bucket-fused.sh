#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-27}"
W=$((N + 1))
ARCH="${ARCH:-native}"
LOW_LUT_K="${LOW_LUT_K:-$((W / 2))}"
HIGH_LUT_K="${HIGH_LUT_K:-$((W - LOW_LUT_K - 1))}"
TRANSPOSE_MODE="${TRANSPOSE_MODE:-events}"

if (( LOW_LUT_K <= 0 || HIGH_LUT_K <= 0 || LOW_LUT_K + HIGH_LUT_K + 1 != W )); then
  echo "invalid factor split" >&2
  exit 2
fi

case "$TRANSPOSE_MODE" in
  sync)
    SRC_REL="src/cuda/b300/oneesan_cuda_gridfp_b300_bucket_fused.cu"
    SUFFIX=""
    ;;
  events)
    SRC_REL="src/cuda/b300/oneesan_cuda_gridfp_b300_bucket_fused_events.cu"
    SUFFIX="_events"
    ;;
  *)
    echo "invalid TRANSPOSE_MODE=$TRANSPOSE_MODE (expected sync or events)" >&2
    exit 2
    ;;
esac

SRC="$(repo_path "$SRC_REL")"
OUT="$(build_path "${OUT:-oneesan_cuda_gridfp_b300_bucket_fused${SUFFIX}_n${N}}")"

TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" \
  -DTARGET_W="$W" -DLOW_LUT_K="$LOW_LUT_K" -DHIGH_LUT_K="$HIGH_LUT_K" \
  "$SRC" -o "$OUT"

echo "built $OUT (transpose=$TRANSPOSE_MODE)"
