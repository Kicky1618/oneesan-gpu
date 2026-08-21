#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
ARCH="${ARCH:-sm_90}"
BATCH="${BATCH:-4}"
OUT="$(build_path "${OUT:-oneesan_cuda_hopper_nc128}")"
TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -arch="$ARCH" -maxrregcount=56 -lineinfo \
  -DRES_BATCH="$BATCH" \
  "$ONEESAN_ROOT/src/cuda/hopper/oneesan_cuda_hopper_nc128.cu" -o "$OUT"
echo "built $OUT (RES_BATCH=$BATCH)"
