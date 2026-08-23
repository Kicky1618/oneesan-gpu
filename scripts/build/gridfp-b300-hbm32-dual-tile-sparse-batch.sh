#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-27}"
W=$((N + 1))
ARCH="${ARCH:-native}"
LOW_LUT_K="${LOW_LUT_K:-$((W / 2))}"
HIGH_LUT_K="${HIGH_LUT_K:-$((W - LOW_LUT_K - 1))}"

if (( LOW_LUT_K <= 0 || HIGH_LUT_K <= 0 || LOW_LUT_K + HIGH_LUT_K + 1 != W )); then
  echo "invalid factor split" >&2
  exit 2
fi

SRC="$(repo_path "src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_dual_tile_sparse_batch.cu")"
OUT="$(build_path "${OUT:-oneesan_cuda_gridfp_b300_hbm32_dual_tile_sparse_batch_n${N}}")"

TMPDIR="$ONEESAN_TMP_DIR" nvcc \
  -O3 -std=c++17 -lineinfo -arch="$ARCH" \
  -DTARGET_W="$W" \
  -DLOW_LUT_K="$LOW_LUT_K" \
  -DHIGH_LUT_K="$HIGH_LUT_K" \
  "$SRC" -o "$OUT"

echo "built $OUT"
echo "  n=$N width=$W arch=$ARCH low_lut_k=$LOW_LUT_K high_lut_k=$HIGH_LUT_K"
echo "  exact: python3 scripts/solve/solve_b300_exact_batch.py $N --binary $OUT"
echo "  env: ONEESAN_DUAL_THREADS=256 ONEESAN_DUAL_CHUNK_MIB=512"
