#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

W="${W:-10}"; ARCH="${ARCH:-sm_80}"; PM_ACCUM="${PM_ACCUM:-0}"; DECODE_LOAD="${DECODE_LOAD:-ldg}"; RANKSTREAM_LUT_LOAD="${RANKSTREAM_LUT_LOAD:-constant}"; RANKCHUNK32_ONESHFL="${RANKCHUNK32_ONESHFL:-1}"; RANKCHUNK32_FUSED16="${RANKCHUNK32_FUSED16:-0}"
LOW_LUT_K="${LOW_LUT_K:-$((W / 2))}"; HIGH_LUT_K="${HIGH_LUT_K:-$((W - LOW_LUT_K - 1))}"
if (( W > 12 || LOW_LUT_K <= 0 || HIGH_LUT_K <= 0 || LOW_LUT_K + HIGH_LUT_K + 1 != W )); then echo "rankchunk32 CROSS5 selftest requires valid W<=12 split" >&2; exit 2; fi
if [[ "$PM_ACCUM" != 0 && "$PM_ACCUM" != 1 ]]; then echo "PM_ACCUM must be 0 or 1" >&2; exit 2; fi
if [[ "$RANKCHUNK32_ONESHFL" != 0 && "$RANKCHUNK32_ONESHFL" != 1 ]]; then echo "RANKCHUNK32_ONESHFL must be 0 or 1" >&2; exit 2; fi
if [[ "$RANKCHUNK32_FUSED16" != 0 && "$RANKCHUNK32_FUSED16" != 1 ]]; then echo "RANKCHUNK32_FUSED16 must be 0 or 1" >&2; exit 2; fi
case "$DECODE_LOAD" in global) P10DC_DECODE_LDG=0 ;; ldg) P10DC_DECODE_LDG=1 ;; *) echo "DECODE_LOAD must be global or ldg" >&2; exit 2;; esac
P10DC_RANKSTREAM_LUT_LDG=0
P10DC_RANKSTREAM_LUT_PAD256=0
case "$RANKSTREAM_LUT_LOAD" in
  constant) ;;
  ldg) P10DC_RANKSTREAM_LUT_LDG=1 ;;
  ldg256) P10DC_RANKSTREAM_LUT_LDG=1; P10DC_RANKSTREAM_LUT_PAD256=1 ;;
  *) echo "RANKSTREAM_LUT_LOAD must be constant, ldg, or ldg256" >&2; exit 2;;
esac
SRC="$ONEESAN_ROOT/src/cuda/gridfp/probes/ramstream32_bucket_orbit_closure_pattern10_depthcode_rankchunk32_cross5_selftest.cu"
BIN="${BIN:-$ONEESAN_BUILD_DIR/pattern10_depthcode_rankchunk32_cross5_selftest_w${W}_pm${PM_ACCUM}_${DECODE_LOAD}_ranklut${RANKSTREAM_LUT_LOAD}_oneshfl${RANKCHUNK32_ONESHFL}_fused16${RANKCHUNK32_FUSED16}}"
mkdir -p "$(dirname "$BIN")"
TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" \
  -DTARGET_W="$W" -DLOW_LUT_K="$LOW_LUT_K" -DHIGH_LUT_K="$HIGH_LUT_K" \
  -DGPU_DIRECT_PM_ACCUM="$PM_ACCUM" -DP10DC_DECODE_LDG="$P10DC_DECODE_LDG" \
  -DP10DC_RANKSTREAM_LUT_LDG="$P10DC_RANKSTREAM_LUT_LDG" \
  -DP10DC_RANKSTREAM_LUT_PAD256="$P10DC_RANKSTREAM_LUT_PAD256" \
  -DP10DC_RANKCHUNK32_ONESHFL="$RANKCHUNK32_ONESHFL" \
  -DP10DC_RANKCHUNK32_FUSED16="$RANKCHUNK32_FUSED16" \
  "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Eq "bucket-closure-pattern10-depthcode-rankchunk32-cross5-selftest (OK W=$W|SKIP no CUDA device)" <<<"$out"
if grep -Fq "OK W=$W" <<<"$out"; then
  grep -Fq 'forward_exact=1 reverse_exact=1 rankchunk32_table_exact=1 padding_exact=1' <<<"$out"
  grep -Fq 'chunk_bits=24 prefix_bits=8 block=16 height_align=32' <<<"$out"
  grep -Fq 'block_base_loads_per_warp_max=2' <<<"$out"
  grep -Fq 'cross_runtime_div=0 cross_runtime_mod=0 cross_runtime_direct_lookup=0' <<<"$out"
  grep -Fq 'old_prekey_offset_arrays_freed=1 fallback_structurally_unreachable=1' <<<"$out"
fi
echo "pattern10-depthcode-rankchunk32-cross5-selftest OK W=$W pm_accum=$PM_ACCUM decode_load=$DECODE_LOAD rankstream_lut_load=$RANKSTREAM_LUT_LOAD aligned_meta=1 rankchunk32_oneshfl=$RANKCHUNK32_ONESHFL rankchunk32_fused16=$RANKCHUNK32_FUSED16" >&2
