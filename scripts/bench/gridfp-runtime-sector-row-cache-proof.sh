#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"; command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_runtime_sector_row_cache_proof.cpp"; BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_runtime_sector_row_cache_proof}"; mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"; out="$($BIN)"; printf '%s\n' "$out"
grep -Fq 'gridfp-runtime-sector-row-cache-proof OK' <<<"$out"
grep -Fq 'W_configs=11 table_entries=1199 context_bytes=24' <<<"$out"
grep -Fq 'max_outer=13 max_row_base=975 uint16_exact=1 contiguous_exact=1 footprint_unchanged=1' <<<"$out"
echo 'gridfp-runtime-sector-row-cache-proof OK footprint_unchanged=1' >&2
