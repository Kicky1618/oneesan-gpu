#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"
command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/b300_high_drop_chunk_proof.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/b300_high_drop_chunk_proof}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'b300-high-drop-chunk-proof OK' <<<"$out"
grep -Fq 'p_range=15..27 max_chunks=3 max_tail=3 signed56_safe=1 height_carry_exact=1 scalar_delta_exact=1' <<<"$out"
echo 'b300-high-drop-chunk-proof OK exact=1' >&2
