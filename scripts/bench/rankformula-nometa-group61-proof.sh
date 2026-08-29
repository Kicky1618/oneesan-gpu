#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_rankformula_nometa_group61.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_rankformula_nometa_group61}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'gridfp-rankformula-nometa-group61 OK groups=69632' <<<"$out"
grep -Fq 'max_start=29113 start_bits=15 max_end=30114 end_bits=15' <<<"$out"
grep -Fq 'max_source_base=29113 source_base_bits=15 max_lcount=7 lcount_bits=3' <<<"$out"
grep -Fq 'max_abstract_off=7059 abstract_off_bits=13 max_height_rank=30114' <<<"$out"
grep -Fq 'packed_bits=61 spare_bits=3 exact=69632 direct_end_compare=1 direct_source_base=1 signed_delta_decode=0' <<<"$out"
echo 'rankformula-nometa-group61-proof OK packed61=1 end15=1 source_base15=1 exact=69632' >&2
