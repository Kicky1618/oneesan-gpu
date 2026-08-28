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
grep -Fq 'packed_bytes=56480 overflow_states=429 overflow_bytes=858 total_bytes=57338' <<<"$out"
grep -Fq 'first6_bits=60 source_bits=10 overflow_only_n14_h0=1' <<<"$out"
echo 'rankformula-abstract-srcpack10-proof OK bytes=57338 exact=32743' >&2
