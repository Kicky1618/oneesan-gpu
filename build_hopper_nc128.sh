#!/usr/bin/env bash
set -euo pipefail
mkdir -p .tmp
BATCH="${BATCH:-4}"
OUT="${OUT:-oneesan_cuda_hopper_nc128}"
TMPDIR="$PWD/.tmp" nvcc -O3 -std=c++17 -arch=sm_90 -maxrregcount=56 -lineinfo \
  -DRES_BATCH="$BATCH" oneesan_cuda_hopper_nc128.cu -o "$OUT"
echo "built $OUT (RES_BATCH=$BATCH)"
