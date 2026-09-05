#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

CXX="${CXX:-g++}"
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_rankformula_slotmeta.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_rankformula_slotmeta}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'gridfp-rankformula-slotmeta OK' <<<"$out"
grep -Fq 'codes=1201917 exact=1201917' <<<"$out"
grep -Fq 'max_owned_masks=2050 max_slot=2049' <<<"$out"
grep -Fq 'slot_bits=12 lmask_bits=14 metadata_bits=26 metadata_bytes=4' <<<"$out"
grep -Fq 'max_reverse_slot_mask_bytes=4100' <<<"$out"
grep -Fq 'old_direct_mask_slot_bytes=32768' <<<"$out"
grep -Fq 'broadword_support=0 direct_mask_slot=0' <<<"$out"
echo 'rankformula-slotmeta-proof OK exact_w28=1 metadata_bits=26 reverse_mask_max=4100B' >&2
