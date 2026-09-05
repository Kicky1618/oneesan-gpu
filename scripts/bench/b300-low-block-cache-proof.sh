#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"
command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/b300_low_block_cache_proof.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/b300_low_block_cache_proof}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'b300-low-block-cache-proof OK' <<<"$out"
grep -Fq 'packed_bits=64' <<<"$out"
grep -Fq 'extra_per_state_bytes=0' <<<"$out"
grep -Fq 'exact=1' <<<"$out"
echo 'b300-low-block-cache-proof OK exact=1' >&2
