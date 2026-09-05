#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_rankformula_abstract_select8.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_rankformula_abstract_select8}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'gridfp-rankformula-abstract-select8 OK' <<<"$out"
grep -Fq 'abstract_states=7060 select_depths=13 table_entries=91780 table_bytes=91780' <<<"$out"
grep -Fq 'universal_nonzero=12755 universal_selected=19273' <<<"$out"
grep -Fq 'production_entries=15624921 production_nonzero=1757173 production_selected=2492769' <<<"$out"
grep -Fq 'max_lcount=7 depth14_selected=0 depth15_selected=0 exact_rankmask=1 select_bits=7' <<<"$out"
echo 'rankformula-abstract-select8-proof OK bytes=91780 exact_rankmask=1' >&2
