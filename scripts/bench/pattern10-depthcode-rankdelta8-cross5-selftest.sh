#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

W="${W:-10}"; ARCH="${ARCH:-sm_80}"; PM_ACCUM="${PM_ACCUM:-0}"
DECODE_LOAD="${DECODE_LOAD:-ldg}"; RANKSTREAM_LUT_LOAD="${RANKSTREAM_LUT_LOAD:-ldg}"
RANKCHUNK32_FUSED16="${RANKCHUNK32_FUSED16:-1}"; RANKDELTA8_ALIGN32="${RANKDELTA8_ALIGN32:-1}"
LOW_LUT_K="${LOW_LUT_K:-$((W / 2))}"; HIGH_LUT_K="${HIGH_LUT_K:-$((W - LOW_LUT_K - 1))}"
if (( W > 12 || LOW_LUT_K <= 0 || HIGH_LUT_K <= 0 || LOW_LUT_K + HIGH_LUT_K + 1 != W )); then echo "rankdelta8 selftest requires valid W<=12 split" >&2; exit 2; fi
for x in PM_ACCUM RANKCHUNK32_FUSED16 RANKDELTA8_ALIGN32; do v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0 or 1" >&2; exit 2; }; done
case "$DECODE_LOAD" in global) P10DC_DECODE_LDG=0 ;; ldg) P10DC_DECODE_LDG=1 ;; *) echo "DECODE_LOAD must be global or ldg" >&2; exit 2;; esac
P10DC_RANKSTREAM_LUT_LDG=0; P10DC_RANKSTREAM_LUT_PAD256=0
case "$RANKSTREAM_LUT_LOAD" in
  constant) ;;
  ldg) P10DC_RANKSTREAM_LUT_LDG=1 ;;
  ldg256) P10DC_RANKSTREAM_LUT_LDG=1; P10DC_RANKSTREAM_LUT_PAD256=1 ;;
  *) echo "RANKSTREAM_LUT_LOAD must be constant, ldg, or ldg256" >&2; exit 2;;
esac

SRC="$ONEESAN_ROOT/src/cuda/gridfp/probes/ramstream32_bucket_orbit_closure_pattern10_depthcode_rankdelta8_cross5_selftest.cu"
BIN="${BIN:-$ONEESAN_BUILD_DIR/pattern10_depthcode_rankdelta8_cross5_selftest_w${W}_pm${PM_ACCUM}_${DECODE_LOAD}_ranklut${RANKSTREAM_LUT_LOAD}_fused${RANKCHUNK32_FUSED16}_align${RANKDELTA8_ALIGN32}}"
mkdir -p "$(dirname "$BIN")"
TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" \
  -DTARGET_W="$W" -DLOW_LUT_K="$LOW_LUT_K" -DHIGH_LUT_K="$HIGH_LUT_K" \
  -DGPU_DIRECT_PM_ACCUM="$PM_ACCUM" -DP10DC_DECODE_LDG="$P10DC_DECODE_LDG" \
  -DP10DC_RANKSTREAM_LUT_LDG="$P10DC_RANKSTREAM_LUT_LDG" \
  -DP10DC_RANKSTREAM_LUT_PAD256="$P10DC_RANKSTREAM_LUT_PAD256" \
  -DP10DC_RANKCHUNK32_FUSED16="$RANKCHUNK32_FUSED16" \
  -DP10DC_RANKCHUNK32_BYTEPACK=0 -DP10DC_RANKCHUNK32_BLOCK64=0 \
  -DP10DC_RANKDELTA8_ALIGN32="$RANKDELTA8_ALIGN32" \
  "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Eq "bucket-closure-pattern10-depthcode-rankdelta8-cross5-selftest (OK W=$W|SKIP no CUDA device)" <<<"$out"
if grep -Fq "OK W=$W" <<<"$out"; then
  grep -Fq 'forward_exact=1 reverse_exact=1 table_exact=1 stream_exact=1' <<<"$out"
  grep -Fq 'chunk_bits=23 prefix_bits=9 block=32' <<<"$out"
  grep -Fq 'delta_varint=7_or_14_bits' <<<"$out"
  grep -Fq 'cross_runtime_div=0 cross_runtime_mod=0' <<<"$out"
fi
echo "pattern10-depthcode-rankdelta8-cross5-selftest OK W=$W pm=$PM_ACCUM lut=$RANKSTREAM_LUT_LOAD fused16=$RANKCHUNK32_FUSED16 align32=$RANKDELTA8_ALIGN32" >&2
