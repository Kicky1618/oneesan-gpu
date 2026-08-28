#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-27}"; W=$((N + 1))
LOW_LUT_K="${LOW_LUT_K:-$((W / 2))}"; HIGH_LUT_K="${HIGH_LUT_K:-$((W - LOW_LUT_K - 1))}"
CXX="${CXX:-g++}"
if (( LOW_LUT_K <= 0 || HIGH_LUT_K <= 0 || LOW_LUT_K + HIGH_LUT_K + 1 != W )); then echo "invalid factor split" >&2; exit 2; fi
if (( LOW_LUT_K > 14 || HIGH_LUT_K > 14 )); then echo "ternary-delta proof requires half widths <=14" >&2; exit 2; fi
if ! command -v "$CXX" >/dev/null; then echo "$CXX not found" >&2; exit 2; fi
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_closure_ternary_delta_phase.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_closure_ternary_delta_phase_w${W}}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 -I"$ONEESAN_ROOT/src/common" \
  -DTARGET_W="$W" -DLOW_LUT_K="$LOW_LUT_K" -DHIGH_LUT_K="$HIGH_LUT_K" \
  "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq "gridfp-closure-ternary-delta-phase OK W=$W" <<<"$out"
grep -Fq 'phase_complete=1 payload_only=1 mateid_source_rebuild_required=0' <<<"$out"
for phase in forward_low reverse_low forward_high reverse_high; do
  grep -Eq "${phase}_max_sources=[1-8]" <<<"$out"
done
echo "closure-ternary-delta-proof OK W=$W low=$LOW_LUT_K high=$HIGH_LUT_K phases=forward_low,reverse_low,forward_high,reverse_high" >&2
