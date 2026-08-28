#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-27}"
W=$((N + 1))
MODE="${SHARD_ADDRESS_MODE:-0}"
ARCH="${ARCH:-native}"
LOW_LUT_K="${LOW_LUT_K:-}"
HIGH_LUT_K="${HIGH_LUT_K:-}"
SRC="$(repo_path "${SRC:-src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_fullmate_dropN.cu}")"
GENERATOR="$ONEESAN_ROOT/scripts/build/gen-b300-shard-address-mode.py"
GENSRC="${GENSRC:-$ONEESAN_BUILD_DIR/generated_b300_hbm32_n${N}_shardmode${MODE}.cu}"
OUT="$(build_path "${OUT:-oneesan_cuda_gridfp_b300_hbm32_n${N}_shardmode${MODE}}")"

if [[ "$MODE" != 0 && "$MODE" != 1 && "$MODE" != 2 ]]; then
  echo "SHARD_ADDRESS_MODE must be 0, 1, or 2" >&2
  exit 2
fi
if [[ "$MODE" == 2 && "$N" != 27 ]]; then
  echo "SHARD_ADDRESS_MODE=2 is specialized for n=27 / W=28" >&2
  exit 2
fi
if [[ -z "$LOW_LUT_K" ]]; then
  if (( N >= 27 )); then LOW_LUT_K=13; else LOW_LUT_K=0; fi
fi
if [[ -z "$HIGH_LUT_K" ]]; then
  if (( N >= 27 )); then HIGH_LUT_K=13; else HIGH_LUT_K=0; fi
fi

if [[ "$MODE" == 2 ]]; then
  bash "$ONEESAN_ROOT/scripts/bench/b300-shard-owner-mulhi-w28-ngpu8-proof.sh"
fi
python3 "$GENERATOR" "$SRC" "$GENSRC"

TMPDIR="$ONEESAN_TMP_DIR" nvcc \
  -O3 -std=c++17 -lineinfo \
  -arch="$ARCH" \
  -DTARGET_W="$W" \
  -DLOW_LUT_K="$LOW_LUT_K" \
  -DHIGH_LUT_K="$HIGH_LUT_K" \
  -DB300_FAST_SHARD_ADDRESS8=0 \
  -DB300_SHARD_ADDRESS_MODE="$MODE" \
  "$GENSRC" -o "$OUT"

echo "built $OUT"
echo "  source=$SRC"
echo "  generated_source=$GENSRC"
echo "  n=$N width=$W arch=$ARCH low_lut_k=$LOW_LUT_K high_lut_k=$HIGH_LUT_K shard_address_mode=$MODE"
echo "  mode0=runtime_u64_division"
echo "  mode1=three_compare_subtract_stages"
echo "  mode2=w28x8_mulhi_masked_base"
