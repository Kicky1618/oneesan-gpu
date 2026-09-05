#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_rankformula_nometa4_group64.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_rankformula_nometa4_group64}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'gridfp-rankformula-nometa4-group64 OK' <<<"$out"
grep -Fq 'codes=1201917 groups=69632 blocks=300524' <<<"$out"
grep -Fq 'group_bytes=557056 block_bytes=601048 total_bytes=1158104' <<<"$out"
grep -Fq 'max_start=29113 max_group_count=1001 max_owner_groups=8709 max_group_index=8708 max_support_count=14' <<<"$out"
grep -Fq 'min_base_delta=-12969 max_base_delta=14873 max_locator_steps=3' <<<"$out"
grep -Fq 'packed_bits=59 spare_bits=5 self_index_exact=1 count_pack_exact=1 support_count_pack_exact=1 support_position_bits=0 group_n_bits=4 sentinel_entries=0' <<<"$out"
echo 'rankformula-nometa4-group64-proof OK total_bytes=1158104 packed59=1 selfindex=1 max_steps=3' >&2
