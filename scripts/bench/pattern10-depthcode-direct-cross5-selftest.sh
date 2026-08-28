#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

W="${W:-10}"; ARCH="${ARCH:-sm_80}"; PM_ACCUM="${PM_ACCUM:-0}"; DECODE_LOAD="${DECODE_LOAD:-ldg}"
LOW_LUT_K="${LOW_LUT_K:-$((W / 2))}"; HIGH_LUT_K="${HIGH_LUT_K:-$((W - LOW_LUT_K - 1))}"
if (( W > 12 || LOW_LUT_K <= 0 || HIGH_LUT_K <= 0 || LOW_LUT_K + HIGH_LUT_K + 1 != W )); then echo "direct CROSS5 selftest requires valid W<=12 split" >&2; exit 2; fi
if [[ "$PM_ACCUM" != 0 && "$PM_ACCUM" != 1 ]]; then echo "PM_ACCUM must be 0 or 1" >&2; exit 2; fi
case "$DECODE_LOAD" in global) P10DC_DECODE_LDG=0 ;; ldg) P10DC_DECODE_LDG=1 ;; *) echo "DECODE_LOAD must be global or ldg" >&2; exit 2;; esac
if ! command -v nvcc >/dev/null; then echo "nvcc not found" >&2; exit 2; fi
SRC="$ONEESAN_ROOT/src/cuda/gridfp/probes/ramstream32_bucket_orbit_closure_pattern10_depthcode_direct_cross5_selftest.cu"
BIN="${BIN:-$ONEESAN_BUILD_DIR/pattern10_depthcode_direct_cross5_selftest_w${W}_pm${PM_ACCUM}_${DECODE_LOAD}}"
mkdir -p "$(dirname "$BIN")"
TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" \
  -DTARGET_W="$W" -DLOW_LUT_K="$LOW_LUT_K" -DHIGH_LUT_K="$HIGH_LUT_K" \
  -DGPU_DIRECT_PM_ACCUM="$PM_ACCUM" -DP10DC_DECODE_LDG="$P10DC_DECODE_LDG" \
  "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Eq "bucket-closure-pattern10-depthcode-direct-cross5-selftest (OK W=$W|SKIP no CUDA device)" <<<"$out"
if grep -Fq "OK W=$W" <<<"$out"; then
  grep -Fq 'control=resolved experiment=warpstriped_delta_direct_cross5' <<<"$out"
  grep -Fq 'forward_exact=1 reverse_exact=1' <<<"$out"
  grep -Fq 'cross5_table_bytes=6561' <<<"$out"
  grep -Fq 'intermediate_plan_local_descriptors=0' <<<"$out"
  grep -Fq "pm_accum=$PM_ACCUM" <<<"$out"
fi
echo "pattern10-depthcode-direct-cross5-selftest OK W=$W pm_accum=$PM_ACCUM decode_load=$DECODE_LOAD" >&2
