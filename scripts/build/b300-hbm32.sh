#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-27}"
W=$((N + 1))
ARCH="${ARCH:-native}"
SRC="$(repo_path "${SRC:-src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_fullmate_dropN.cu}")"
OUT="$(build_path "${OUT:-oneesan_cuda_gridfp_b300_hbm32_n${N}}")"
LOW_LUT_K="${LOW_LUT_K:-}"
HIGH_LUT_K="${HIGH_LUT_K:-}"
FAST_SHARD_ADDRESS8="${FAST_SHARD_ADDRESS8:-0}"
if [[ "$FAST_SHARD_ADDRESS8" != 0 && "$FAST_SHARD_ADDRESS8" != 1 ]]; then
  echo "FAST_SHARD_ADDRESS8 must be 0 or 1" >&2
  exit 2
fi

if [[ -z "$LOW_LUT_K" ]]; then
  if (( N >= 27 )); then LOW_LUT_K=13; else LOW_LUT_K=0; fi
fi
if [[ -z "$HIGH_LUT_K" ]]; then
  if (( N >= 27 )); then HIGH_LUT_K=13; else HIGH_LUT_K=0; fi
fi

if [[ "$FAST_SHARD_ADDRESS8" == 1 ]]; then
  bash "$ONEESAN_ROOT/scripts/bench/b300-shard-address8-proof.sh"
fi

TMPDIR="$ONEESAN_TMP_DIR" nvcc \
  -O3 -std=c++17 -lineinfo \
  -arch="$ARCH" \
  -DTARGET_W="$W" \
  -DLOW_LUT_K="$LOW_LUT_K" \
  -DHIGH_LUT_K="$HIGH_LUT_K" \
  -DB300_FAST_SHARD_ADDRESS8="$FAST_SHARD_ADDRESS8" \
  "$SRC" -o "$OUT"

echo "built $OUT"
echo "  source=$SRC"
echo "  n=$N width=$W arch=$ARCH low_lut_k=$LOW_LUT_K high_lut_k=$HIGH_LUT_K fast_shard_address8=$FAST_SHARD_ADDRESS8"
