#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"
command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/b300_saturation_constant_budget_proof.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/b300_saturation_constant_budget_proof}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'b300-saturation-constant-budget-proof OK' <<<"$out"
grep -Fq 'hot_step_abs_max=1060346729' <<<"$out"
grep -Fq 'hot_pair_abs_max=1881935601' <<<"$out"
grep -Fq 'int32_exact=1' <<<"$out"
grep -Fq 'hot_delta_bytes=17400 closure_table_bytes=20880' <<<"$out"
grep -Fq 'total_constant_bytes=59392 constant_headroom_bytes=6144 exact=1' <<<"$out"
echo 'b300-saturation-constant-budget-proof OK total=59392 headroom=6144 exact=1' >&2
