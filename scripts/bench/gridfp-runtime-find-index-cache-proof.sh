#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"; command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_runtime_find_index_cache_proof.cpp"; BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_runtime_find_index_cache_proof}"; mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"; out="$($BIN)"; printf '%s\n' "$out"
grep -Fq 'gridfp-runtime-find-index-cache-proof OK' <<<"$out"
for buckets in 16 32 64; do grep -Fq "buckets=$buckets components=100000 exact_queries=3200000" <<<"$out"; done
grep -Fq 'bucket_counts=16,32,64 max_pairs=20 total_exact_queries=9600000' <<<"$out"
grep -Fq 'stale_bytes_clear_required=0 false_negative=0 exact=1' <<<"$out"
echo 'gridfp-runtime-find-index-cache-proof OK exact=1' >&2
