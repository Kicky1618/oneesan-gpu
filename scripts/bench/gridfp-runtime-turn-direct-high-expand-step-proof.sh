#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"; command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_runtime_turn_direct_high_expand_step_proof.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_runtime_turn_direct_high_expand_step_proof}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"; printf '%s\n' "$out"
grep -Fq 'gridfp-runtime-turn-direct-high-expand-step-proof OK' <<<"$out"
grep -Fq 'exhaustive_width_max=12' <<<"$out"
grep -Fq 'random_width=28 random_cases=1000000' <<<"$out"
grep -Fq 'source_scope=main_only forward_p=Wm1' <<<"$out"
grep -Fq 'include_result_dispatch=0' <<<"$out"
grep -Fq 'generic_project_forward_calls=0' <<<"$out"
grep -Fq 'full_mirror_passes=0 step_exact=1' <<<"$out"
echo 'gridfp-runtime-turn-direct-high-expand-step-proof OK exact=1' >&2
