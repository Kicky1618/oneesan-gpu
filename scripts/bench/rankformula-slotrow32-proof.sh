#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_rankformula_slotrow32.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_rankformula_slotrow32}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'gridfp-rankformula-slotrow32 OK' <<<"$out"
grep -Fq 'codes=1201917' <<<"$out"
grep -Fq 'packed_bits=30 word_bytes=4' <<<"$out"
grep -Fq 'max_slot=2049' <<<"$out"
grep -Fq 'loads_per_lookup=1 separate_slot_support=0' <<<"$out"
echo 'rankformula-slotrow32-proof OK packed30=1 load1=1' >&2
