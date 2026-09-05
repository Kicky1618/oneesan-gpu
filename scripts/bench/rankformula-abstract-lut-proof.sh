#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_rankformula_abstract_lut.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_rankformula_abstract_lut}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'gridfp-rankformula-abstract-lut OK' <<<"$out"
grep -Fq 'production_codes=1201917 production_transitions=3720805' <<<"$out"
grep -Fq 'abstract_states=7060 abstract_transitions=32743' <<<"$out"
grep -Fq 'descriptor_bytes=28240 source_rank_bytes=65486 offset_bytes=480 total_lut_bytes=94206' <<<"$out"
grep -Fq 'max_source_local_rank=1000' <<<"$out"
grep -Fq 'mask_position_independent=1 all_production_codes_exact=1 all_production_transitions_exact=1' <<<"$out"
echo 'rankformula-abstract-lut-proof OK lut_bytes=94206 states=7060 transitions=32743' >&2
