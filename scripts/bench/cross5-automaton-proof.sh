#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"
if ! command -v "$CXX" >/dev/null; then echo "$CXX not found" >&2; exit 2; fi
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_cross5_automaton.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_cross5_automaton}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'gridfp-cross5-automaton OK' <<<"$out"
grep -Fq 'table_bytes=15552' <<<"$out"
grep -Fq 'max_factor=14 max_chunks=3' <<<"$out"
grep -Fq 'candidate_mask_exact=1 state_exact=1 halt_exact=1 metadata_per_orbit=0' <<<"$out"
echo 'cross5-automaton-proof OK table_bytes=15552 max_chunks=3 metadata_per_orbit=0' >&2
