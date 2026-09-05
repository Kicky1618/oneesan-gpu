#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"
command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/b300_high_main_recurrence_proof.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/b300_high_main_recurrence_proof}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'b300-high-main-recurrence-proof OK' <<<"$out"
grep -Fq 'min_fixed=7 fallback_fixed_lt=7 p_lo=15 symbol_lo=14 symbol_hi=27' <<<"$out"
grep -Fq 'signed35=1 trit_positions=14 trit_chunks=5 trit_bits=25 height_bits=4 total_bits=64' <<<"$out"
grep -Fq 'exact=1' <<<"$out"
echo 'b300-high-main-recurrence-proof OK gated_fixed_bits=7 p_lo=15 trits=14 exact=1' >&2
