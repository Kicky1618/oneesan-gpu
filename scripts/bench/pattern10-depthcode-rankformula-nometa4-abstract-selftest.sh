#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

W="${W:-10}"; ARCH="${ARCH:-sm_80}"; PM_ACCUM="${PM_ACCUM:-0}"
DECODE_LOAD="${DECODE_LOAD:-ldg}"; RANKSTREAM_LUT_LOAD="${RANKSTREAM_LUT_LOAD:-ldg}"
LOW_LUT_K="${LOW_LUT_K:-$((W / 2))}"; HIGH_LUT_K="${HIGH_LUT_K:-$((W - LOW_LUT_K - 1))}"
if (( W > 12 || LOW_LUT_K <= 0 || HIGH_LUT_K <= 0 || LOW_LUT_K + HIGH_LUT_K + 1 != W )); then exit 2; fi
case "$DECODE_LOAD" in global) P10DC_DECODE_LDG=0 ;; ldg) P10DC_DECODE_LDG=1 ;; *) exit 2;; esac
P10DC_RANKSTREAM_LUT_LDG=0; P10DC_RANKSTREAM_LUT_PAD256=0
case "$RANKSTREAM_LUT_LOAD" in constant) ;; ldg) P10DC_RANKSTREAM_LUT_LDG=1 ;; ldg256) P10DC_RANKSTREAM_LUT_LDG=1; P10DC_RANKSTREAM_LUT_PAD256=1 ;; *) exit 2;; esac
SRC="$ONEESAN_ROOT/src/cuda/gridfp/probes/ramstream32_bucket_orbit_closure_pattern10_depthcode_rankformula_nometa4_abstract_selftest.cu"
BIN="${BIN:-$ONEESAN_BUILD_DIR/pattern10_depthcode_rankformula_nometa4_abstract_selftest_w${W}}"
mkdir -p "$(dirname "$BIN")"
TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" \
  -DTARGET_W="$W" -DLOW_LUT_K="$LOW_LUT_K" -DHIGH_LUT_K="$HIGH_LUT_K" \
  -DGPU_DIRECT_PM_ACCUM="$PM_ACCUM" -DP10DC_DECODE_LDG="$P10DC_DECODE_LDG" \
  -DP10DC_RANKSTREAM_LUT_LDG="$P10DC_RANKSTREAM_LUT_LDG" -DP10DC_RANKSTREAM_LUT_PAD256="$P10DC_RANKSTREAM_LUT_PAD256" \
  -DP10DC_RANKCHUNK32_ONESHFL=1 -DP10DC_RANKCHUNK32_FUSED16=0 -DP10DC_RANKCHUNK32_BYTEPACK=0 \
  -DP10DC_RANKCHUNK32_ALIGN32=0 -DP10DC_RANKCHUNK32_BLOCK64=0 -DP10DC_RANKDELTA8_FUSED13=1 \
  -DP10DC_RANKFORMULA_SPARSE_BASE=1 -DP10DC_RANKFORMULA_RAWCODE=1 -DP10DC_RANKFORMULA_INLINE_CROSS=1 \
  -DP10DC_RANKFORMULA_BASE_DELTA=0 -DP10DC_RANKFORMULA_SLOTMETA=0 -DP10DC_RANKFORMULA_SLOTROW32=0 \
  "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Eq "bucket-closure-pattern10-depthcode-rankformula-nometa4-abstract-selftest (OK W=$W|SKIP no CUDA device)" <<<"$out"
if grep -Fq "OK W=$W" <<<"$out"; then
  grep -Fq 'forward_exact=1 reverse_exact=1' <<<"$out"
  grep -Fq 'per_code_metadata_bytes=0 block_size=4 max_locator_steps=3' <<<"$out"
  grep -Fq 'abstract_lut_bytes=94206 abstract_states=7060 abstract_transitions=32743' <<<"$out"
  grep -Fq 'ballot_runtime_loads=0 source_local_lut=1' <<<"$out"
fi
echo "pattern10-depthcode-rankformula-nometa4-abstract-selftest OK W=$W lut=$RANKSTREAM_LUT_LOAD" >&2
