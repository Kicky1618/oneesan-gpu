#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"
command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/b300_low_main_recurrence_proof.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/b300_low_main_recurrence_proof}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'b300-low-main-recurrence-proof OK' <<<"$out"
grep -Fq 'signed31=1 trit_chunks=5 trit_bits=25 height_bits=4 total_bits=60' <<<"$out"
grep -Fq 'exact=1' <<<"$out"
echo 'b300-low-main-recurrence-proof OK exact=1' >&2
