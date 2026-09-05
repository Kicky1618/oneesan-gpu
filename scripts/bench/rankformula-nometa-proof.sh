#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_rankformula_nometa.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_rankformula_nometa}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'gridfp-rankformula-nometa OK' <<<"$out"
grep -Fq 'codes=1201917 unrank_exact=1201917' <<<"$out"
grep -Fq 'groups=69632 block32=37628' <<<"$out"
grep -Fq 'group_table_bytes_all=278528 block_table_bytes_all=75256 aux_bytes_all=353784 old_meta_bytes_all=4807668' <<<"$out"
grep -Fq 'max_groups_owner_height=1027 max_groups_per_block=32 max_locator_steps=31' <<<"$out"
grep -Fq 'per_code_metadata_bytes=0 ballot_unrank_exact=1' <<<"$out"
echo 'rankformula-nometa-proof OK exact_unrank=1 aux_bytes=353784 old_meta_bytes=4807668 max_locator_steps=31' >&2
