#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"
if ! command -v "$CXX" >/dev/null; then echo "$CXX not found" >&2; exit 2; fi
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_rankchunk32_bytepack.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_rankchunk32_bytepack}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'gridfp-rankchunk32-bytepack OK' <<<"$out"
grep -Fq 'ternary_keys=4782969' <<<"$out"
grep -Fq 'max_l_per_legal_code=7' <<<"$out"
grep -Fq 'max_prefix=217 prefix8_exact=1' <<<"$out"
grep -Fq 'chunk_bits=24 prefix_bits=8 byte_aligned_chunks=1' <<<"$out"
grep -Fq 'max_third_chunk=80 bit23_always_zero=1 pack_roundtrip_exact=1' <<<"$out"
echo 'rankchunk32-bytepack-proof OK K=14 block=32 max_prefix=217' >&2
