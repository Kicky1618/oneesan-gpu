#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_rankformula_nometa_blocks.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_rankformula_nometa_blocks}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'block=4 codes=1201917 groups=69632 blocks=300524 group_bytes=557056 block_bytes=601048 aux_bytes=1158104 max_groups_per_block=4 max_locator_steps=3' <<<"$out"
grep -Fq 'block=8 codes=1201917 groups=69632 blocks=150293 group_bytes=557056 block_bytes=300586 aux_bytes=857642 max_groups_per_block=8 max_locator_steps=7' <<<"$out"
grep -Fq 'block=16 codes=1201917 groups=69632 blocks=75175 group_bytes=557056 block_bytes=150350 aux_bytes=707406 max_groups_per_block=16 max_locator_steps=15' <<<"$out"
grep -Fq 'block=32 codes=1201917 groups=69632 blocks=37628 group_bytes=557056 block_bytes=75256 aux_bytes=632312 max_groups_per_block=32 max_locator_steps=31' <<<"$out"
grep -Fq 'gridfp-rankformula-nometa-blocks OK old_meta_bytes=4807668 group_entry_bytes=8 b4_aux_bytes=1158104 b8_aux_bytes=857642 b4_max_steps=3 b8_max_steps=7 ab_blocks=4,8' <<<"$out"
echo 'rankformula-nometa-blocks-proof OK group64=1 ab_blocks=4,8' >&2
