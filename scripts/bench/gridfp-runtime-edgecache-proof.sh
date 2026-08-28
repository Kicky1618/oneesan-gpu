#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

CXX="${CXX:-g++}"
if ! command -v "$CXX" >/dev/null; then
  echo "$CXX not found" >&2
  exit 2
fi

SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_runtime_edgecache_proof.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_runtime_edgecache_proof}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"

grep -Fq 'gridfp-runtime-edgecache-proof OK' <<<"$out"
grep -Fq 'coefficient_merge_cases=31 packing_cases=80 routing_cases=210 accumulation_cases=400' <<<"$out"
grep -Fq 'subgroup_width=8 max_pairs=20 max_edge_terms=3' <<<"$out"
grep -Fq 'coefficient_range=-1..2' <<<"$out"
grep -Fq 'max_destination_slot=2' <<<"$out"
grep -Fq 'cache_bytes_per_subgroup=80 cache_bytes_per_block=2560' <<<"$out"
grep -Fq 'coefficient_bound_exact=1 packing_exact=1 routing_exact=1 accumulation_exact=1' <<<"$out"

echo 'gridfp-runtime-edgecache-proof OK coefficient_merge_cases=31 packing_cases=80 routing_cases=210 accumulation_cases=400 cache_bytes_per_block=2560' >&2
