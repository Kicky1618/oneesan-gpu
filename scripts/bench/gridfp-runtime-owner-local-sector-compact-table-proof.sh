#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"
command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_runtime_owner_local_sector_compact_table_proof.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_runtime_owner_local_sector_compact_table_proof}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'gridfp-runtime-owner-local-sector-compact-table-proof OK' <<<"$out"
grep -Fq 'full_entries=1100 compact_entries=503 full_bytes=4400 compact_bytes=2012 saved_bytes=2388' <<<"$out"
grep -Fq 'max_end=448876754 W28_base=405 positive_exact=1 formula_exact=1' <<<"$out"
echo 'gridfp-runtime-owner-local-sector-compact-table-proof OK exact=1' >&2
