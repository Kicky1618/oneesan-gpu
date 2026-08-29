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
MAIN_MATE_CACHE="${MAIN_MATE_CACHE:-1}"
PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"

for name in FAST_SHARD_ADDRESS8 MAIN_MATE_CACHE PTXAS_VERBOSE; do
  value="${!name}"
  if [[ "$value" != 0 && "$value" != 1 ]]; then
    echo "$name must be 0 or 1" >&2
    exit 2
  fi
done

# The previous production default enabled the rank/unrank LUTs only for n=27.
# On B300 this leaves n=24..26 dominated by integer DP walks instead of HBM
# traffic.  K=13 is still small compared with B300 HBM and removes roughly half
# of every width-25..28 rank/unrank walk, so use it throughout the production
# range.  Smaller test widths retain the zero-LUT baseline unless overridden.
if [[ -z "$LOW_LUT_K" ]]; then
  if (( N >= 24 )); then LOW_LUT_K=13; else LOW_LUT_K=0; fi
fi
if [[ -z "$HIGH_LUT_K" ]]; then
  if (( N >= 24 )); then HIGH_LUT_K=13; else HIGH_LUT_K=0; fi
fi
if (( LOW_LUT_K > W || HIGH_LUT_K > W - 1 )); then
  echo "LUT K exceeds width: n=$N W=$W low=$LOW_LUT_K high=$HIGH_LUT_K" >&2
  exit 2
fi

if [[ "$FAST_SHARD_ADDRESS8" == 1 ]]; then
  bash "$ONEESAN_ROOT/scripts/bench/b300-shard-address8-proof.sh"
fi

BUILD_SRC="$SRC"
if [[ "$MAIN_MATE_CACHE" == 1 ]]; then
  BUILD_SRC="$ONEESAN_BUILD_DIR/b300_hbm32_n${N}_main_mate_cache.cu"
  python3 "$ONEESAN_ROOT/scripts/build/gen-b300-main-mate-cache.py" "$SRC" "$BUILD_SRC"
fi

PTXAS_FLAGS=()
if [[ "$PTXAS_VERBOSE" == 1 ]]; then
  PTXAS_FLAGS+=("-Xptxas=-v")
fi

TMPDIR="$ONEESAN_TMP_DIR" nvcc \
  -O3 -std=c++17 -lineinfo \
  -arch="$ARCH" \
  "${PTXAS_FLAGS[@]}" \
  -DTARGET_W="$W" \
  -DLOW_LUT_K="$LOW_LUT_K" \
  -DHIGH_LUT_K="$HIGH_LUT_K" \
  -DB300_FAST_SHARD_ADDRESS8="$FAST_SHARD_ADDRESS8" \
  "$BUILD_SRC" -o "$OUT"

echo "built $OUT"
echo "  source=$SRC"
echo "  build_source=$BUILD_SRC"
echo "  n=$N width=$W arch=$ARCH low_lut_k=$LOW_LUT_K high_lut_k=$HIGH_LUT_K fast_shard_address8=$FAST_SHARD_ADDRESS8 main_mate_cache=$MAIN_MATE_CACHE ptxas_verbose=$PTXAS_VERBOSE"
