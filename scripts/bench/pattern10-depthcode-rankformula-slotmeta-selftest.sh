#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

W="${W:-10}"; ARCH="${ARCH:-sm_80}"; PM_ACCUM="${PM_ACCUM:-0}"
DECODE_LOAD="${DECODE_LOAD:-ldg}"; RANKSTREAM_LUT_LOAD="${RANKSTREAM_LUT_LOAD:-ldg}"
RANKDELTA8_FUSED13="${RANKDELTA8_FUSED13:-1}"
LOW_LUT_K="${LOW_LUT_K:-$((W / 2))}"; HIGH_LUT_K="${HIGH_LUT_K:-$((W - LOW_LUT_K - 1))}"
if (( W > 12 || LOW_LUT_K <= 0 || HIGH_LUT_K <= 0 || LOW_LUT_K + HIGH_LUT_K + 1 != W )); then
  echo "rankformula slotmeta selftest requires valid W<=12 split" >&2; exit 2
fi
for x in PM_ACCUM RANKDELTA8_FUSED13; do
  v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0 or 1" >&2; exit 2; }
done
case "$DECODE_LOAD" in global) P10DC_DECODE_LDG=0 ;; ldg) P10DC_DECODE_LDG=1 ;; *) exit 2;; esac
P10DC_RANKSTREAM_LUT_LDG=0; P10DC_RANKSTREAM_LUT_PAD256=0
case "$RANKSTREAM_LUT_LOAD" in
  constant) ;;
  ldg) P10DC_RANKSTREAM_LUT_LDG=1 ;;
  ldg256) P10DC_RANKSTREAM_LUT_LDG=1; P10DC_RANKSTREAM_LUT_PAD256=1 ;;
  *) exit 2;;
esac

SRC="$ONEESAN_ROOT/src/cuda/gridfp/probes/ramstream32_bucket_orbit_closure_pattern10_depthcode_rankformula_cross5_selftest.cu"
BIN="${BIN:-$ONEESAN_BUILD_DIR/pattern10_depthcode_rankformula_slotmeta_selftest_w${W}_pm${PM_ACCUM}_${DECODE_LOAD}_ranklut${RANKSTREAM_LUT_LOAD}}"
mkdir -p "$(dirname "$BIN")"
TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" \
  -DTARGET_W="$W" -DLOW_LUT_K="$LOW_LUT_K" -DHIGH_LUT_K="$HIGH_LUT_K" \
  -DGPU_DIRECT_PM_ACCUM="$PM_ACCUM" -DP10DC_DECODE_LDG="$P10DC_DECODE_LDG" \
  -DP10DC_RANKSTREAM_LUT_LDG="$P10DC_RANKSTREAM_LUT_LDG" \
  -DP10DC_RANKSTREAM_LUT_PAD256="$P10DC_RANKSTREAM_LUT_PAD256" \
  -DP10DC_RANKCHUNK32_FUSED16=0 -DP10DC_RANKCHUNK32_BYTEPACK=0 -DP10DC_RANKCHUNK32_BLOCK64=0 \
  -DP10DC_RANKDELTA8_FUSED13="$RANKDELTA8_FUSED13" \
  -DP10DC_RANKFORMULA_SPARSE_BASE=1 -DP10DC_RANKFORMULA_RAWCODE=0 \
  -DP10DC_RANKFORMULA_INLINE_CROSS=1 -DP10DC_RANKFORMULA_BASE_DELTA=1 \
  -DP10DC_RANKFORMULA_SLOTMETA=1 \
  "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Eq "bucket-closure-pattern10-depthcode-rankformula-cross5-selftest (OK W=$W|SKIP no CUDA device)" <<<"$out"
if grep -Fq "OK W=$W" <<<"$out"; then
  grep -Fq 'forward_exact=1 reverse_exact=1' <<<"$out"
  grep -Fq 'sparse_base=1 base_delta=1 base_values_per_lookup=1 rawcode=0 slotmeta=1' <<<"$out"
  grep -Fq 'chunkinfo_elided=1 broadword_support_elided=1 direct_mask_slot_elided=1' <<<"$out"
  grep -Fq 'inline_cross=1 cross_lut_elided=1' <<<"$out"
  grep -Fq 'cross_runtime_div=0 cross_runtime_mod=0' <<<"$out"
fi
echo "pattern10-depthcode-rankformula-slotmeta-selftest OK W=$W pm=$PM_ACCUM lut=$RANKSTREAM_LUT_LOAD" >&2
