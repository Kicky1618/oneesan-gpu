#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

CXX="${CXX:-g++}"
command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_runtime_turn_local_sector_w28_tree_proof.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_runtime_turn_local_sector_w28_tree_proof}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'gridfp-runtime-turn-local-sector-w28-tree-proof OK' <<<"$out"
grep -Fq 'W=28 rows=14 count7_rows=7 count8_rows=7 entries=105' <<<"$out"
grep -Fq 'final_endpoint_is_group=1' <<<"$out"
grep -Fq 'max_generic_comparisons=4 max_generic_loads=5 max_tree_loads=3' <<<"$out"
grep -Fq 'begin_reload_eliminated=1 exact=1' <<<"$out"
echo 'gridfp-runtime-turn-local-sector-w28-tree-proof OK exact=1' >&2
