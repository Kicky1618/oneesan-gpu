#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ARCH="${ARCH:-sm_90}"
TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -arch="$ARCH" -lineinfo \
  "$ONEESAN_ROOT/src/cuda/gridfp/oneesan_cuda_gridfp_parallel.cu" \
  -o "$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_parallel_hopper"
TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -arch="$ARCH" -lineinfo \
  "$ONEESAN_ROOT/src/cuda/gridfp/oneesan_cuda_gridfp_window.cu" \
  -o "$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_window_hopper"
