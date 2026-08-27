#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
N="${N:-27}"; W=$((N + 1)); ARCH="${ARCH:-sm_80}"
LOW_LUT_K="${LOW_LUT_K:-$((W / 2))}"; HIGH_LUT_K="${HIGH_LUT_K:-$((W - LOW_LUT_K - 1))}"
if (( LOW_LUT_K <= 0 || HIGH_LUT_K <= 0 || LOW_LUT_K + HIGH_LUT_K + 1 != W )); then echo "invalid factor split" >&2; exit 2; fi
if (( LOW_LUT_K > 14 || HIGH_LUT_K > 14 )); then echo "pattern10 depthcode plan requires half widths <=14" >&2; exit 2; fi
SRC="$ONEESAN_ROOT/src/cuda/gridfp/probes/ramstream32_pattern10_depthcode_build_plan.cu"
BIN="${BIN:-$ONEESAN_BUILD_DIR/pattern10_depthcode_build_plan_w${W}}"
mkdir -p "$(dirname "$BIN")"
TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" \
  -DTARGET_W="$W" -DLOW_LUT_K="$LOW_LUT_K" -DHIGH_LUT_K="$HIGH_LUT_K" "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq "pattern10-depthcode-build-plan OK W=$W" <<<"$out"
grep -Fq 'sidecar_bytes_per_orbit=0' <<<"$out"
grep -Fq 'payload_exact=1' <<<"$out"
grep -Fq 'decode_payload_masks=1' <<<"$out"
grep -Fq 'decode_unrank=0' <<<"$out"
grep -Fq 'temporary_depth_bytes=0' <<<"$out"
grep -Fq 'builder_passes=1' <<<"$out"
grep -Fq 'dense_context_bitset_bytes=0' <<<"$out"
echo "pattern10-depthcode-build-plan OK n=$N payload=predecoded sidecar=0 builder_passes=1 dense_bitset=0" >&2
