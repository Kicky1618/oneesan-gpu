#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_rankformula_nometa_group56_coop.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_rankformula_nometa_group56_coop}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'gridfp-rankformula-nometa-group56-coop OK codes=1201917 blocks=75175 exact=1201917' <<<"$out"
grep -Fq 'cooperative_successor_loads=65159 max_steps=15 max_abstract_off=7059' <<<"$out"
grep -Fq 'block=16 leader_gi_increment_exact=1 packed56_decode_exact=1 off_table_load_required=0' <<<"$out"
echo 'rankformula-nometa-group56-coop-proof OK exact=1201917 successor_loads=65159' >&2
