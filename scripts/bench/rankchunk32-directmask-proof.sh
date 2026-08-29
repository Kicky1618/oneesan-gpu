#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

CXX="${CXX:-g++}"
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_rankchunk32_directmask_proof.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_rankchunk32_directmask_proof}"
mkdir -p "$(dirname "$BIN")"

"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'gridfp-rankchunk32-directmask-proof OK' <<<"$out"
grep -Fq 'k_range=1..14' <<<"$out"
grep -Fq 'depth_range=1..15' <<<"$out"
grep -Fq 'mask_or=0x7f' <<<"$out"
grep -Fq 'chunk_composition_exact=1' <<<"$out"
grep -Fq 'partial_chunk_exact=1' <<<"$out"
grep -Fq 'global_rank_ordinal_exact=1' <<<"$out"
grep -Fq 'max_mask_bits=7' <<<"$out"
echo "rankchunk32-directmask-proof OK" >&2
