#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_rankformula_nometa_group56.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_rankformula_nometa_group56}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'gridfp-rankformula-nometa-group56 OK groups=69632' <<<"$out"
grep -Fq 'max_start=29113 start_bits=15 max_lcount=7 lcount_bits=3' <<<"$out"
grep -Fq 'min_delta=-12969 max_delta=14873 delta_signed_bits=15' <<<"$out"
grep -Fq 'max_count=1001 count_bits=10 max_abstract_off=7059 abstract_off_bits=13' <<<"$out"
grep -Fq 'packed_bits=56 spare_bits=8 exact=69632' <<<"$out"
grep -Fq 'n_reconstructed_from_h_lcount=1 self_group_index_removed=1 coop_leader_gi_register=1' <<<"$out"
echo 'rankformula-nometa-group56-proof OK packed56=1 abstract_off13=1 exact=69632' >&2
