#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

CXX="${CXX:-g++}"
BIN="${BIN:-$ONEESAN_BUILD_DIR/rankformula_directgather64_proof}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$ONEESAN_ROOT/src/cpp/probes/gridfp_rankformula_directgather64.cpp" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'rankformula-directgather64-proof' <<<"$out"
grep -Fq 'production_descriptors=15624921' <<<"$out"
grep -Fq 'selected_sources=2492769' <<<"$out"
grep -Fq 'count0=13867748' <<<"$out"
grep -Fq 'rare_ge4=26545' <<<"$out"
grep -Fq 'source_rank_bits=15 rare_index_bits=16' <<<"$out"
grep -Fq 'exact=1' <<<"$out"
echo "rankformula-directgather64-proof OK" >&2
