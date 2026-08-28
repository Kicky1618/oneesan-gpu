#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"
if ! command -v "$CXX" >/dev/null; then echo "$CXX not found" >&2; exit 2; fi
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_rankchunk32_block64.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_rankchunk32_block64}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'gridfp-rankchunk32-block64 OK' <<<"$out"
grep -Fq 'block=64 prefix_bits=9' <<<"$out"
grep -Fq 'max_l_per_legal_code=7 max_prefix=441' <<<"$out"
grep -Fq 'crossing_alignments_full_warp=31 noncrossing_alignments_full_warp=33' <<<"$out"
grep -Fq 'block_table_reduction_vs32=2x direct_block_exact=1' <<<"$out"
echo 'rankchunk32-block64-proof OK block=64 max_prefix=441' >&2
