#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_rankformula_nometa_group61_coop.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_rankformula_nometa_group61_coop}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'gridfp-rankformula-nometa-group61-coop OK codes=1201917 blocks=75175 exact=1201917' <<<"$out"
grep -Fq 'cooperative_successor_loads=65159 max_steps=15 max_end=30114 max_source_base=29113' <<<"$out"
grep -Fq 'block=16 leader_gi_increment_exact=1 direct_end_compare_exact=1 direct_source_base_exact=1' <<<"$out"
echo 'rankformula-nometa-group61-coop-proof OK exact=1201917 successor_loads=65159 direct=1' >&2
