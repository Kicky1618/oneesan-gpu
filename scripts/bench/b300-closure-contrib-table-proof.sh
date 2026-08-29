#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"
command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/b300_closure_contrib_table_proof.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/b300_closure_contrib_table_proof}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'b300-closure-contrib-table-proof OK' <<<"$out"
grep -Fq 'semantics=lexicographic_prefix_mass exact=1' <<<"$out"
grep -Fq 'table_bytes=13920' <<<"$out"
grep -Fq 'known_constant_bytes=34800' <<<"$out"
echo 'b300-closure-contrib-table-proof OK exact=1 constant_budget=OK' >&2
