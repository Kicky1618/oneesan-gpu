#!/usr/bin/env bash
set -euo pipefail

N="${N:-27}"
W=$((N + 1))
ARCH="${ARCH:-native}"
SRC="${SRC:-oneesan_cuda_gridfp_b300_hbm32_fullmate_dropN.cu}"
OUT="${OUT:-oneesan_cuda_gridfp_b300_hbm32_n${N}}"
LOW_LUT_K="${LOW_LUT_K:-}"
HIGH_LUT_K="${HIGH_LUT_K:-}"

# The n=27 / 16-GiB scratch plan fixes 13 frontier occupancy bits per
# window.  LOW/HIGH=13 cover the complete fixed side.  The scratch arena
# opportunistically adds a full MateID cache only when the group still fits TARGET_MIB.
if [[ -z "$LOW_LUT_K" ]]; then
  if (( N >= 27 )); then LOW_LUT_K=13; else LOW_LUT_K=0; fi
fi
if [[ -z "$HIGH_LUT_K" ]]; then
  if (( N >= 27 )); then HIGH_LUT_K=13; else HIGH_LUT_K=0; fi
fi

mkdir -p .tmp
TMPDIR="$PWD/.tmp" nvcc \
  -O3 -std=c++17 -lineinfo \
  -arch="$ARCH" \
  -DTARGET_W="$W" \
  -DLOW_LUT_K="$LOW_LUT_K" \
  -DHIGH_LUT_K="$HIGH_LUT_K" \
  "$SRC" -o "$OUT"

echo "built $OUT from $SRC for n=$N width=$W arch=$ARCH low_lut_k=$LOW_LUT_K high_lut_k=$HIGH_LUT_K"
