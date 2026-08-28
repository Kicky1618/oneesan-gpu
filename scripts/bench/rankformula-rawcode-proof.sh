#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

CXX="${CXX:-g++}"
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_rankformula_rawcode.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_rankformula_rawcode}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'gridfp-rankformula-rawcode OK' <<<"$out"
grep -Fq 'codes=4782969' <<<"$out"
grep -Fq 'support_exact=4782969' <<<"$out"
grep -Fq 'chunks_exact=4782969' <<<"$out"
grep -Fq 'max_chunk_key=242' <<<"$out"
grep -Fq 'metadata_bits=28 metadata_bytes=4' <<<"$out"
grep -Fq 'chunkinfo_loads=0' <<<"$out"
grep -Fq 'runtime_div=0 runtime_mod=0' <<<"$out"
echo 'rankformula-rawcode-proof OK exact_3pow14=1 metadata_bytes=4 chunkinfo_loads=0' >&2
