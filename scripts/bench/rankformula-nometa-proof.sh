#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_rankformula_nometa.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_rankformula_nometa}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'gridfp-rankformula-nometa OK' <<<"$out"
grep -Fq 'codes=1201917 unrank_exact=1201917' <<<"$out"
grep -Fq 'per_code_metadata_bytes=0 ballot_unrank_exact=1' <<<"$out"
echo 'rankformula-nometa-proof OK exact_unrank=1 per_code_metadata_bytes=0' >&2
