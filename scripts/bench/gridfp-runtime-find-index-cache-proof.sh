#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"; command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_runtime_find_index_cache_proof.cpp"; BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_runtime_find_index_cache_proof}"; mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"; out="$($BIN)"; printf '%s\n' "$out"
grep -Fq 'gridfp-runtime-find-index-cache-proof OK' <<<"$out"
for cfg in 'storage_bytes=16 ways=1 hash_buckets=16' 'storage_bytes=32 ways=1 hash_buckets=32' 'storage_bytes=64 ways=1 hash_buckets=64' 'storage_bytes=32 ways=2 hash_buckets=16' 'storage_bytes=64 ways=2 hash_buckets=32' 'storage_bytes=64 ways=4 hash_buckets=16'; do
  grep -Fq "$cfg" <<<"$out"
done
grep -Fq 'configs=6 max_pairs=20 total_exact_queries=9600000' <<<"$out"
grep -Fq 'set_associative=1 overflow_state=packed_high_bit' <<<"$out"
grep -Fq 'extra_overflow_registers=0' <<<"$out"
grep -Fq 'stale_bytes_clear_required=0 false_negative=0 exact=1' <<<"$out"
echo 'gridfp-runtime-find-index-cache-proof OK exact=1 set_associative=1' >&2
