#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

NVCC="${NVCC:-nvcc}"
N="${N:-27}"
W=$((N + 1))
ARCH="${ARCH:-native}"
LOW_LUT_K="${LOW_LUT_K:-13}"
HIGH_LUT_K="${HIGH_LUT_K:-13}"
SRC="$(repo_path "${SRC:-src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_fullmate_dropN.cu}")"
GENERATOR="$ONEESAN_ROOT/scripts/build/gen-b300-shard-address-hi32-seed.py"
GENSRC="${GENSRC:-$ONEESAN_BUILD_DIR/generated_b300_hbm32_n${N}_shardmode5.cu}"
OUT="$(build_path "${OUT:-oneesan_cuda_gridfp_b300_hbm32_n${N}_shardmode5}")"

if [[ "$N" != 27 ]]; then
  echo "hi32 seed shard address is specialized for n=27 / W=28" >&2
  exit 2
fi
command -v "$NVCC" >/dev/null || { echo "$NVCC not found" >&2; exit 2; }
bash "$ONEESAN_ROOT/scripts/bench/b300-shard-owner-hi32-seed-w28-ngpu8-proof.sh"
python3 "$GENERATOR" "$SRC" "$GENSRC"

TMPDIR="$ONEESAN_TMP_DIR" "$NVCC" \
  -O3 -std=c++17 -lineinfo -arch="$ARCH" \
  -DTARGET_W="$W" \
  -DLOW_LUT_K="$LOW_LUT_K" \
  -DHIGH_LUT_K="$HIGH_LUT_K" \
  -DB300_FAST_SHARD_ADDRESS8=0 \
  -DB300_SHARD_ADDRESS_MODE=5 \
  "$GENSRC" -o "$OUT"

echo "built $OUT"
echo "  source=$SRC"
echo "  generated_source=$GENSRC"
echo "  n=$N width=$W arch=$ARCH low_lut_k=$LOW_LUT_K high_lut_k=$HIGH_LUT_K shard_address_mode=5"
echo "  mode5=w28x8_hi32_seed_single_correction"
