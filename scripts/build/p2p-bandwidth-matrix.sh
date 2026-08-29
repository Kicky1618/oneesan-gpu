#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ARCH="${ARCH:-native}"
SRC="$(repo_path "src/cuda/gridfp/oneesan_p2p_bandwidth_matrix.cu")"
OUT="$(build_path "${OUT:-oneesan_p2p_bandwidth_matrix}")"

TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" "$SRC" -o "$OUT"
echo "built $OUT"
echo "  run: eval \"\$($OUT 256 8)\""
echo "  then: N=22 scripts/build/gridfp-gpu-direct-atomicfree-multigpu-topomap.sh"
