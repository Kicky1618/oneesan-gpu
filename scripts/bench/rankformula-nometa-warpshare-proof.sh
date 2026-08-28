#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_rankformula_nometa_warpshare.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_rankformula_nometa_warpshare}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'block=4 codes=1201917 blocks=300524 locator_steps=104346 scalar_table_loads=2508180 warpshare_table_loads=705394' <<<"$out"
grep -Fq 'block=8 codes=1201917 blocks=150293 locator_steps=243417 scalar_table_loads=2647251 warpshare_table_loads=544003' <<<"$out"
grep -Fq 'block=16 codes=1201917 blocks=75175 locator_steps=521034 scalar_table_loads=2924868 warpshare_table_loads=671384' <<<"$out"
grep -Fq 'gridfp-rankformula-nometa-warpshare OK prefix_active_widths=32 stripe_bases=4 blocks=4,8,16 best_block=8 block8_shared_loads=544003' <<<"$out"
grep -Fq 'representative_lane_active=1 same_block=1' <<<"$out"
echo 'rankformula-nometa-warpshare-proof OK best_block=8 shared_loads=544003' >&2
