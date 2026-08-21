#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ARCH="${ARCH:-native}"
SRC="$(repo_path "${SRC:-src/cuda/gridfp/oneesan_cuda_gridfp_multigpu_mmap.cu}")"
OUT="$(build_path "${OUT:-oneesan_cuda_gridfp_multigpu_mmap}")"

TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -arch="$ARCH" -lineinfo "$SRC" -o "$OUT"
echo "built $OUT from $SRC for $ARCH"
