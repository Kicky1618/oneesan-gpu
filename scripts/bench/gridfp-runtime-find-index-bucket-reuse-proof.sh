#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"; command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_runtime_find_index_bucket_reuse_proof.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_runtime_find_index_bucket_reuse_proof}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"; printf '%s\n' "$out"
grep -Fq 'gridfp-runtime-find-index-bucket-reuse-proof OK' <<<"$out"
grep -Fq 'hash_buckets=16' <<<"$out"
grep -Fq 'hash_buckets=32' <<<"$out"
grep -Fq 'hash_buckets=64' <<<"$out"
grep -Fq 'memo_storage=unused_occupancy_high_bits' <<<"$out"
grep -Fq 'hash64_fallback=rehash shared_bytes_added=0 exact=1' <<<"$out"
grep -Fq 'occupancy_low_bits_exact=1 record_bucket_exact=1' <<<"$out"
echo 'gridfp-runtime-find-index-bucket-reuse-proof OK exact=1 shared_bytes_added=0' >&2
