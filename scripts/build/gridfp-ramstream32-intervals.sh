#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ARCH="${ARCH:-native}"
TARGET_W="${TARGET_W:-28}"
MIN_INTERVAL_ELEMS="${MIN_INTERVAL_ELEMS:-16384}"
SRC="$(repo_path "${SRC:-src/cuda/gridfp/oneesan_cuda_gridfp_ramstream32_intervals.cu}")"
OUT="$(build_path "${OUT:-oneesan_cuda_gridfp_ramstream32_intervals_w${TARGET_W}}")"

TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -arch="$ARCH" -lineinfo \
  -DTARGET_W="$TARGET_W" \
  -DRAMSTREAM_MIN_INTERVAL_ELEMS="$MIN_INTERVAL_ELEMS" \
  "$SRC" -o "$OUT"

echo "built $OUT from $SRC for $ARCH TARGET_W=$TARGET_W MIN_INTERVAL_ELEMS=$MIN_INTERVAL_ELEMS"
