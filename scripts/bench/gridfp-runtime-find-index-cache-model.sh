#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"; MAX_W="${MAX_W:-10}"
command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
if (( MAX_W < 5 || MAX_W > 12 )); then echo "MAX_W must be 5..12" >&2; exit 2; fi
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_runtime_find_index_cache_model.cpp"; BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_runtime_find_index_cache_model}"; mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN "$MAX_W")"; printf '%s\n' "$out"
grep -Fq 'ALL_OK runtime_find_index_cache_model=1' <<<"$out"
grep -Fq 'collision_state=packed_high_bit' <<<"$out"
grep -Fq 'extra_collision_registers=0' <<<"$out"
grep -Fq 'stale_bytes_guarded_by_occupancy=1' <<<"$out"
echo "gridfp-runtime-find-index-cache-model OK maxW=$MAX_W collision_state=packed_high_bit" >&2
