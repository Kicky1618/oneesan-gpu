#!/usr/bin/env bash
set -euo pipefail
mkdir -p .tmp
ARCH="${ARCH:-sm_90}"
TMPDIR="$PWD/.tmp" nvcc -O3 -std=c++17 -arch="$ARCH" -lineinfo \
  oneesan_cuda_gridfp_parallel.cu -o oneesan_cuda_gridfp_parallel_hopper
TMPDIR="$PWD/.tmp" nvcc -O3 -std=c++17 -arch="$ARCH" -lineinfo \
  oneesan_cuda_gridfp_window.cu -o oneesan_cuda_gridfp_window_hopper
