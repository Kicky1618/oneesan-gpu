#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"
command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_runtime_turn_local_sector_carry_begin_proof.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_runtime_turn_local_sector_carry_begin_proof}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'gridfp-runtime-turn-local-sector-carry-begin-proof OK' <<<"$out"
grep -Fq 'production_W_max=28 max_old_table_loads=5 max_new_table_loads=4 begin_reload_eliminated=1 exact=1' <<<"$out"
echo 'gridfp-runtime-turn-local-sector-carry-begin-proof OK exact=1' >&2
