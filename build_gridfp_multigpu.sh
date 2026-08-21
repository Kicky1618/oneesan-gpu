#!/usr/bin/env bash
set -euo pipefail
mkdir -p .tmp
ARCH="${ARCH:-native}"
SRC="${SRC:-oneesan_cuda_gridfp_multigpu_mmap.cu}"
OUT="${OUT:-oneesan_cuda_gridfp_multigpu}"
TMPDIR="$PWD/.tmp" nvcc -O3 -std=c++17 -arch="$ARCH" -lineinfo "$SRC" -o "$OUT"
echo "built $OUT from $SRC for $ARCH"
