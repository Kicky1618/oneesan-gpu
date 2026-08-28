#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

CXX="${CXX:-g++}"
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_rankdelta8_plan.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_rankdelta8_plan}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'gridfp-rankdelta8-plan OK' <<<"$out"
grep -Fq 'max_local_rank=30113' <<<"$out"
grep -Fq 'max_owner_ranks=467761' <<<"$out"
grep -Fq 'max_owner_bytes=619944' <<<"$out"
grep -Fq 'max_delta=572' <<<"$out"
grep -Fq 'original_rankstream_bytes=7441610' <<<"$out"
grep -Fq 'rankdelta8_bytes=4925378' <<<"$out"
grep -Fq 'max_prefix_packed=333' <<<"$out"
grep -Fq 'max_prefix_align32=333' <<<"$out"
echo 'rankdelta8-plan OK exact_w28=1 delta_varint=7_or_14_bits prefix9_block32=1' >&2
