#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_rankformula_abstract_depth4.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_rankformula_abstract_depth4}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'gridfp-rankformula-abstract-depth4 OK' <<<"$out"
grep -Fq 'states=7060 exact_depth_entries=91780 table_bytes=28240 nonzero_depth_slots=19273' <<<"$out"
grep -Fq 'max_selected_depth=13 bits_per_l_ordinal=4 packed_bits_per_state=28' <<<"$out"
grep -Fq 'swar_exact=1 depth14_15_zero=1' <<<"$out"
echo 'rankformula-abstract-depth4-proof OK bytes=28240 exact=91780' >&2
