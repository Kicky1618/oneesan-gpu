#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"
command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_component_support_adjacent_marks_proof.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_component_support_adjacent_marks_proof}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'gridfp-component-support-adjacent-marks-proof OK' <<<"$out"
grep -Fq 'exhaustive_cases=294900' <<<"$out"
grep -Fq 'adjacent_exact=1 lexicographic_exact=1' <<<"$out"
echo 'gridfp-component-support-adjacent-marks-proof OK exact=1' >&2
