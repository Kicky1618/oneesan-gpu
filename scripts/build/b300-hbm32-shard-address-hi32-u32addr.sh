#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
NVCC="${NVCC:-nvcc}"; N="${N:-27}"; W=$((N+1)); ARCH="${ARCH:-native}"
[[ "$N" == 27 ]] || { echo "fully-u32 hi32 shard mode is specialized for n=27" >&2; exit 2; }
command -v "$NVCC" >/dev/null || exit 2
SRC="$(repo_path "${SRC:-src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_fullmate_dropN.cu}")"
GEN="$ONEESAN_ROOT/scripts/build/gen-b300-shard-address-hi32-u32addr.py"
GENSRC="${GENSRC:-$ONEESAN_BUILD_DIR/generated_b300_hbm32_n27_shardmode6.cu}"
OUT="$(build_path "${OUT:-oneesan_cuda_gridfp_b300_hbm32_n27_shardmode6}")"
bash "$ONEESAN_ROOT/scripts/bench/b300-shard-owner-hi32-u32addr-w28-ngpu8-proof.sh"
python3 "$GEN" "$SRC" "$GENSRC"
TMPDIR="$ONEESAN_TMP_DIR" "$NVCC" -O3 -std=c++17 -lineinfo -arch="$ARCH" \
  -DTARGET_W=28 -DLOW_LUT_K="${LOW_LUT_K:-13}" -DHIGH_LUT_K="${HIGH_LUT_K:-13}" \
  -DB300_FAST_SHARD_ADDRESS8=0 -DB300_SHARD_ADDRESS_MODE=6 "$GENSRC" -o "$OUT"
echo "built $OUT"
echo "  generated_source=$GENSRC mode6=w28x8_hi32_seed_fully_u32_address"
