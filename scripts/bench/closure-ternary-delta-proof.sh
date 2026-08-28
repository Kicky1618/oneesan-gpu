#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-27}"; W=$((N + 1))
LOW_LUT_K="${LOW_LUT_K:-$((W / 2))}"; HIGH_LUT_K="${HIGH_LUT_K:-$((W - LOW_LUT_K - 1))}"
CXX="${CXX:-g++}"
if (( LOW_LUT_K <= 0 || HIGH_LUT_K <= 0 || LOW_LUT_K + HIGH_LUT_K + 1 != W )); then echo "invalid factor split" >&2; exit 2; fi
if (( LOW_LUT_K > 14 || HIGH_LUT_K > 14 )); then echo "ternary-delta proof requires half widths <=14" >&2; exit 2; fi
if ! command -v "$CXX" >/dev/null; then echo "$CXX not found" >&2; exit 2; fi
mkdir -p "$ONEESAN_BUILD_DIR"

COMMON=("-O3" "-std=c++17" "-I$ONEESAN_ROOT/src/common" "-DTARGET_W=$W" "-DLOW_LUT_K=$LOW_LUT_K" "-DHIGH_LUT_K=$HIGH_LUT_K")

PHASE_SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_closure_ternary_delta_phase.cpp"
PHASE_BIN="${PHASE_BIN:-$ONEESAN_BUILD_DIR/gridfp_closure_ternary_delta_phase_w${W}}"
"$CXX" "${COMMON[@]}" "$PHASE_SRC" -o "$PHASE_BIN"
phase_out="$($PHASE_BIN)"
printf '%s\n' "$phase_out"
grep -Fq "gridfp-closure-ternary-delta-phase OK W=$W" <<<"$phase_out"
grep -Fq 'phase_complete=1 payload_only=1 mateid_source_rebuild_required=0' <<<"$phase_out"
for phase in forward_low reverse_low forward_high reverse_high; do
  grep -Eq "${phase}_max_sources=[1-8]" <<<"$phase_out"
done

CROSS_SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_closure_ternary_delta_cross.cpp"
CROSS_BIN="${CROSS_BIN:-$ONEESAN_BUILD_DIR/gridfp_closure_ternary_delta_cross_w${W}}"
"$CXX" "${COMMON[@]}" "$CROSS_SRC" -o "$CROSS_BIN"
cross_out="$($CROSS_BIN)"
printf '%s\n' "$cross_out"
grep -Fq "gridfp-closure-ternary-delta-cross OK W=$W" <<<"$cross_out"
grep -Fq 'cross_source_pair_exact=1 cross_delta_exact=1' <<<"$cross_out"

echo "closure-ternary-delta-proof OK W=$W low=$LOW_LUT_K high=$HIGH_LUT_K phases=forward_low,reverse_low,forward_high,reverse_high cross=1" >&2
