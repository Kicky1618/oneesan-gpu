#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_rankformula_abstract_compact.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_rankformula_abstract_compact}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'gridfp-rankformula-abstract-compact OK states=7060 universal_selected=19273 production_selected=2492769' <<<"$out"
grep -Fq 'depth_bytes=28240 source_bytes=57338 total_lut_bytes=86058' <<<"$out"
grep -Fq 'depth4_split16=1 srcpack10_split32=1 exact_selected_sources=1 depth14_15_zero=1' <<<"$out"
echo 'rankformula-abstract-compact-proof OK lut_bytes=86058 exact_selected_sources=1' >&2
