#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_rankformula_abstract_srcpack10.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_rankformula_abstract_srcpack10}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'gridfp-rankformula-abstract-srcpack10 OK' <<<"$out"
grep -Fq 'states=7060 transitions=32743 exact=32743 max_source_local_rank=1000' <<<"$out"
grep -Fq 'packed_word_bytes=56480 overflow_states=429 overflow_bytes=858 total_bytes=57338' <<<"$out"
grep -Fq 'word0_loads=1675973 word1_loads=241173 overflow_loads=132' <<<"$out"
grep -Fq 'dynamic_split_bytes=7668848 dynamic_fixed64_bytes=14057384 dynamic_roffsrc_bytes=8499884' <<<"$out"
grep -Fq 'ranks_per_word=3 source_bits=10 overflow_only_n14_h0=1' <<<"$out"
echo 'rankformula-abstract-srcpack10-proof OK bytes=57338 dynamic_bytes=7668848 exact=32743' >&2
