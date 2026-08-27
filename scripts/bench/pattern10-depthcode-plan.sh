#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-27}"; W=$((N + 1)); ARCH="${ARCH:-sm_80}"
LOW_LUT_K="${LOW_LUT_K:-$((W / 2))}"; HIGH_LUT_K="${HIGH_LUT_K:-$((W - LOW_LUT_K - 1))}"
if (( LOW_LUT_K <= 0 || HIGH_LUT_K <= 0 || LOW_LUT_K + HIGH_LUT_K + 1 != W )); then echo "invalid factor split" >&2; exit 2; fi
if (( LOW_LUT_K > 14 || HIGH_LUT_K > 14 )); then echo "pattern10 depthcode probe requires half widths <=14" >&2; exit 2; fi
SRC="$ONEESAN_ROOT/src/cuda/gridfp/probes/ramstream32_closure_pattern10_depthcode_plan.cu"
BIN="${BIN:-$ONEESAN_BUILD_DIR/pattern10_depthcode_plan_w${W}}"
mkdir -p "$(dirname "$BIN")"
TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" \
  -DTARGET_W="$W" -DLOW_LUT_K="$LOW_LUT_K" -DHIGH_LUT_K="$HIGH_LUT_K" \
  "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq "closure-pattern10-depthcode-plan OK W=$W" <<<"$out"
if grep -Fq 'fits10=1' <<<"$out"; then
  echo "pattern10-depthcode viable: per-context (pattern,depth) pairs fit in the orbit upper 10 bits" >&2
else
  echo "pattern10-depthcode does not fit 10 bits in at least one context; use escape/nibble fallback" >&2
fi
