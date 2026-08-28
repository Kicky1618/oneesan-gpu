#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"; command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_runtime_turn_direct_high_expand_inverse_proof.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_runtime_turn_direct_high_expand_inverse_proof}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"; printf '%s\n' "$out"
grep -Fq 'gridfp-runtime-turn-direct-high-expand-inverse-proof OK' <<<"$out"
grep -Fq 'exhaustive_width_max=12' <<<"$out"
grep -Fq 'random_width=28 random_cases=1000000' <<<"$out"
grep -Fq 'source_scope=main_only forward_p=Wm1' <<<"$out"
grep -Fq 'right_closure_candidates=0' <<<"$out"
grep -Fq 'rl_validity_checks=0' <<<"$out"
grep -Fq 'full_validity_scans_per_candidate=0' <<<"$out"
grep -Fq 'include_rechecks_per_candidate=0' <<<"$out"
grep -Fq 'inverse_set_exact=1' <<<"$out"
echo 'gridfp-runtime-turn-direct-high-expand-inverse-proof OK exact=1' >&2
