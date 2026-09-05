#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

W="${W:-10}"; ARCH="${ARCH:-sm_80}"; PM_ACCUM="${PM_ACCUM:-0}"; DECODE_LOAD="${DECODE_LOAD:-ldg}"; LUT_LOAD="${LUT_LOAD:-constant}"
LOW_LUT_K="${LOW_LUT_K:-$((W / 2))}"; HIGH_LUT_K="${HIGH_LUT_K:-$((W - LOW_LUT_K - 1))}"
if (( W > 12 || LOW_LUT_K <= 0 || HIGH_LUT_K <= 0 || LOW_LUT_K + HIGH_LUT_K + 1 != W )); then echo "rankstream CROSS5 selftest requires valid W<=12 split" >&2; exit 2; fi
if [[ "$PM_ACCUM" != 0 && "$PM_ACCUM" != 1 ]]; then echo "PM_ACCUM must be 0 or 1" >&2; exit 2; fi
case "$DECODE_LOAD" in global) P10DC_DECODE_LDG=0 ;; ldg) P10DC_DECODE_LDG=1 ;; *) echo "DECODE_LOAD must be global or ldg" >&2; exit 2;; esac
P10DC_RANKSTREAM_LUT_LDG=0
P10DC_RANKSTREAM_LUT_PAD256=0
case "$LUT_LOAD" in
  constant) ;;
  ldg) P10DC_RANKSTREAM_LUT_LDG=1 ;;
  ldg256) P10DC_RANKSTREAM_LUT_LDG=1; P10DC_RANKSTREAM_LUT_PAD256=1 ;;
  *) echo "LUT_LOAD must be constant, ldg, or ldg256" >&2; exit 2;;
esac
SRC="$ONEESAN_ROOT/src/cuda/gridfp/probes/ramstream32_bucket_orbit_closure_pattern10_depthcode_rankstream_cross5_selftest.cu"
BIN="${BIN:-$ONEESAN_BUILD_DIR/pattern10_depthcode_rankstream_cross5_selftest_w${W}_pm${PM_ACCUM}_${DECODE_LOAD}_${LUT_LOAD}}"
mkdir -p "$(dirname "$BIN")"
TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" \
  -DTARGET_W="$W" -DLOW_LUT_K="$LOW_LUT_K" -DHIGH_LUT_K="$HIGH_LUT_K" \
  -DGPU_DIRECT_PM_ACCUM="$PM_ACCUM" -DP10DC_DECODE_LDG="$P10DC_DECODE_LDG" \
  -DP10DC_RANKSTREAM_LUT_LDG="$P10DC_RANKSTREAM_LUT_LDG" \
  -DP10DC_RANKSTREAM_LUT_PAD256="$P10DC_RANKSTREAM_LUT_PAD256" \
  "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Eq "bucket-closure-pattern10-depthcode-rankstream-cross5-selftest (OK W=$W|SKIP no CUDA device)" <<<"$out"
if grep -Fq "OK W=$W" <<<"$out"; then
  grep -Fq 'forward_exact=1 reverse_exact=1' <<<"$out"
  grep -Fq 'prekey_scope=fixed_owner prekey_table_exact=1 rankstream_table_exact=1' <<<"$out"
  grep -Fq 'rankstream_model=offset32+rank16_per_L' <<<"$out"
  grep -Fq 'cross_runtime_ternary_fold=0 cross_runtime_direct_lookup=0' <<<"$out"
  grep -Fq 'cross_runtime_ordinal_popcount=0 constant_loads_per_chunk=2' <<<"$out"
  grep -Fq 'fallback_structurally_unreachable=1 rankstream_lut_bytes=6561' <<<"$out"
  grep -Fq 'ordinary_cross5_lut_present=0' <<<"$out"
  grep -Fq "pm_accum=$PM_ACCUM" <<<"$out"
fi
echo "pattern10-depthcode-rankstream-cross5-selftest OK W=$W pm_accum=$PM_ACCUM decode_load=$DECODE_LOAD lut_load=$LUT_LOAD rankstream=1 ordinary_cross5_lut=0" >&2
