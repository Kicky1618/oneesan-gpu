#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"; command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_runtime_find_index_cache_proof.cpp"; BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_runtime_find_index_cache_proof}"; mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"; out="$($BIN)"; printf '%s\n' "$out"
grep -Fq 'gridfp-runtime-find-index-cache-proof OK' <<<"$out"
grep -Fq 'buckets=64 bytes_per_set=64 bytes_per_subgroup=128' <<<"$out"
grep -Fq 'stale_bytes_clear_required=0 false_negative=0 exact=1' <<<"$out"
echo 'gridfp-runtime-find-index-cache-proof OK exact=1' >&2
