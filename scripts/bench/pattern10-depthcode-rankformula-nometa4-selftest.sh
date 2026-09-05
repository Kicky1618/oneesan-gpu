#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

W="${W:-10}"; ARCH="${ARCH:-sm_80}"; PM_ACCUM="${PM_ACCUM:-0}"
DECODE_LOAD="${DECODE_LOAD:-ldg}"; RANKSTREAM_LUT_LOAD="${RANKSTREAM_LUT_LOAD:-ldg}"
LOW_LUT_K="${LOW_LUT_K:-$((W / 2))}"; HIGH_LUT_K="${HIGH_LUT_K:-$((W - LOW_LUT_K - 1))}"
if (( W > 12 || LOW_LUT_K <= 0 || HIGH_LUT_K <= 0 || LOW_LUT_K + HIGH_LUT_K + 1 != W )); then
  echo "rankformula nometa4 selftest requires valid W<=12 split" >&2; exit 2
fi
[[ "$PM_ACCUM" == 0 || "$PM_ACCUM" == 1 ]] || exit 2
case "$DECODE_LOAD" in global) P10DC_DECODE_LDG=0 ;; ldg) P10DC_DECODE_LDG=1 ;; *) exit 2;; esac
P10DC_RANKSTREAM_LUT_LDG=0; P10DC_RANKSTREAM_LUT_PAD256=0
case "$RANKSTREAM_LUT_LOAD" in
  constant) ;;
  ldg) P10DC_RANKSTREAM_LUT_LDG=1 ;;
  ldg256) P10DC_RANKSTREAM_LUT_LDG=1; P10DC_RANKSTREAM_LUT_PAD256=1 ;;
  *) exit 2;;
esac

SRC="$ONEESAN_ROOT/src/cuda/gridfp/probes/ramstream32_bucket_orbit_closure_pattern10_depthcode_rankformula_nometa4_selftest.cu"
BIN="${BIN:-$ONEESAN_BUILD_DIR/pattern10_depthcode_rankformula_nometa4_selftest_w${W}_pm${PM_ACCUM}_${DECODE_LOAD}_${RANKSTREAM_LUT_LOAD}}"
mkdir -p "$(dirname "$BIN")"
TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" \
  -DTARGET_W="$W" -DLOW_LUT_K="$LOW_LUT_K" -DHIGH_LUT_K="$HIGH_LUT_K" \
  -DGPU_DIRECT_PM_ACCUM="$PM_ACCUM" -DP10DC_DECODE_LDG="$P10DC_DECODE_LDG" \
  -DP10DC_RANKSTREAM_LUT_LDG="$P10DC_RANKSTREAM_LUT_LDG" \
  -DP10DC_RANKSTREAM_LUT_PAD256="$P10DC_RANKSTREAM_LUT_PAD256" \
  -DP10DC_RANKCHUNK32_ONESHFL=1 -DP10DC_RANKCHUNK32_FUSED16=0 \
  -DP10DC_RANKCHUNK32_BYTEPACK=0 -DP10DC_RANKCHUNK32_ALIGN32=0 -DP10DC_RANKCHUNK32_BLOCK64=0 \
  -DP10DC_RANKDELTA8_FUSED13=1 \
  -DP10DC_RANKFORMULA_SPARSE_BASE=1 -DP10DC_RANKFORMULA_RAWCODE=1 \
  -DP10DC_RANKFORMULA_INLINE_CROSS=1 -DP10DC_RANKFORMULA_BASE_DELTA=0 \
  -DP10DC_RANKFORMULA_SLOTMETA=0 -DP10DC_RANKFORMULA_SLOTROW32=0 \
  "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Eq "bucket-closure-pattern10-depthcode-rankformula-nometa4-selftest (OK W=$W|SKIP no CUDA device)" <<<"$out"
if grep -Fq "OK W=$W" <<<"$out"; then
  grep -Fq 'forward_exact=1 reverse_exact=1' <<<"$out"
  grep -Fq 'per_code_metadata_bytes=0' <<<"$out"
  grep -Fq 'block_size=4 max_locator_steps=3' <<<"$out"
  grep -Fq 'ballot_unrank=1 formula_ballot=1' <<<"$out"
fi
echo "pattern10-depthcode-rankformula-nometa4-selftest OK W=$W pm=$PM_ACCUM lut=$RANKSTREAM_LUT_LOAD" >&2
