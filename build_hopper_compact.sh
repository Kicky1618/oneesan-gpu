#!/usr/bin/env bash
set -euo pipefail
mkdir -p .tmp
TMPDIR="$PWD/.tmp" nvcc -O3 -std=c++17 -arch=sm_90 -lineinfo \
  oneesan_cuda_hopper_compact.cu -o oneesan_cuda_hopper_compact
