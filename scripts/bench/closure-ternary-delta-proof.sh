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

python3 - "$phase_out" "$cross_out" <<'PY'
import re, sys
phase, cross = sys.argv[1:]
def kv(text):
    return {k:int(v) for k,v in re.findall(r'([a-z_]+)=([0-9]+)', text)}
p=kv(phase); c=kv(cross)
names=('forward_low','reverse_low','forward_high','reverse_high')
states=sum(p[f'{x}_states'] for x in names)
ordinary=sum(p[f'{x}_sources'] for x in names)
crosses=sum(c[f'{x}_cross'] for x in names)
canonical=ordinary+crosses
delta=states
saved=canonical-delta
print(f'ternary_delta_plan_states={states}')
print(f'ternary_delta_canonical_source_key_folds={canonical}')
print(f'ternary_delta_base_key_folds={delta}')
print(f'ternary_delta_source_key_folds_saved={saved}')
print(f'ternary_delta_fold_reduction={canonical/delta:.6f}x' if delta else 'ternary_delta_fold_reduction=NA')
print(f'ternary_delta_saved_fraction={saved/canonical:.9f}' if canonical else 'ternary_delta_saved_fraction=0')
PY

echo "closure-ternary-delta-proof OK W=$W low=$LOW_LUT_K high=$HIGH_LUT_K phases=forward_low,reverse_low,forward_high,reverse_high cross=1 fold_reduction_reported=1" >&2
