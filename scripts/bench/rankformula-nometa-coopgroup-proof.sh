#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_rankformula_nometa_coopgroup.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_rankformula_nometa_coopgroup}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'block=4 codes=1201917 blocks=300524 per_lane_successor_loads=104346 cooperative_successor_loads=52183 cooperative_table_loads=653231' <<<"$out"
grep -Fq 'cooperative_exact=1201917 max_groups_per_block=4' <<<"$out"
grep -Fq 'block=8 codes=1201917 blocks=150293 per_lane_successor_loads=243417 cooperative_successor_loads=60845 cooperative_table_loads=361431' <<<"$out"
grep -Fq 'cooperative_exact=1201917 max_groups_per_block=8' <<<"$out"
grep -Fq 'block=16 codes=1201917 blocks=75175 per_lane_successor_loads=521034 cooperative_successor_loads=65159 cooperative_table_loads=215509' <<<"$out"
grep -Fq 'cooperative_exact=1201917 max_groups_per_block=16' <<<"$out"
grep -Fq 'gridfp-rankformula-nometa-coopgroup OK block8_old_warpshare_loads=544003 block8_cooperative_loads=361431' <<<"$out"
grep -Fq 'fixed_rounds_block8=7 subgroup_prefix_safe=1 cooperative_exact_all=1' <<<"$out"
echo 'rankformula-nometa-coopgroup-proof OK exact=1201917 block8_loads=361431 successor_loads=60845' >&2
