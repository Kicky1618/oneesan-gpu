#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_rankformula_nometa_group64_selfindex.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_rankformula_nometa_group64_selfindex}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'gridfp-rankformula-nometa-group64-selfindex OK groups=69632' <<<"$out"
grep -Fq 'max_start=29113 start_bits=15 max_n=14 n_bits=4' <<<"$out"
grep -Fq 'min_delta=-12969 max_delta=14873 delta_signed_bits=15' <<<"$out"
grep -Fq 'max_count=1001 count_bits=10 max_group_index=8708 group_index_bits=14 max_owner_groups=8709' <<<"$out"
grep -Fq 'packed_bits=59 spare_bits=5 exact=69632 warpshare_gi_shuffle_elidable=1' <<<"$out"
echo 'rankformula-nometa-group64-selfindex-proof OK packed59=1 spare5=1 gi_shuffle_elidable=1' >&2
