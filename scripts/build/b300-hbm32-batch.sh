#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-27}"
W=$((N + 1))
ARCH="${ARCH:-native}"
SRC="${SRC:-}"
if [[ -z "$SRC" ]]; then
  if (( N >= 27 )); then
    SRC="src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_forced2window_opt_batch.cu"
  else
    SRC="src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_fullmate_dropN_batch.cu"
  fi
fi
SRC="$(repo_path "$SRC")"
OUT="$(build_path "${OUT:-oneesan_cuda_gridfp_b300_hbm32_batch_n${N}}")"
LOW_LUT_K="${LOW_LUT_K:-}"
HIGH_LUT_K="${HIGH_LUT_K:-}"

if [[ -z "$LOW_LUT_K" ]]; then
  if (( N >= 27 )); then LOW_LUT_K=14; else LOW_LUT_K=0; fi
fi
if [[ -z "$HIGH_LUT_K" ]]; then
  if (( N >= 27 )); then HIGH_LUT_K=13; else HIGH_LUT_K=0; fi
fi

TMPDIR="$ONEESAN_TMP_DIR" nvcc \
  -O3 -std=c++17 -lineinfo \
  -arch="$ARCH" \
  -DTARGET_W="$W" \
  -DLOW_LUT_K="$LOW_LUT_K" \
  -DHIGH_LUT_K="$HIGH_LUT_K" \
  "$SRC" -o "$OUT"

echo "built $OUT"
echo "  source=$SRC"
echo "  n=$N width=$W arch=$ARCH low_lut_k=$LOW_LUT_K high_lut_k=$HIGH_LUT_K"
