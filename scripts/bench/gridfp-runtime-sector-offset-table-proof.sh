#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"
if ! command -v "$CXX" >/dev/null; then echo "$CXX not found" >&2; exit 2; fi
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_runtime_sector_offset_table_proof.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_runtime_sector_offset_table_proof}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'gridfp-runtime-sector-offset-table-proof OK' <<<"$out"
grep -Fq 'W_configs=11 active_entries=1199 table_entries=1199 table_bytes=4796' <<<"$out"
grep -Fq 'max_offset=1805186805 row_bases_exact=1 uint32_exact=1 embedded_exact=1' <<<"$out"
echo 'gridfp-runtime-sector-offset-table-proof OK embedded_exact=1' >&2
