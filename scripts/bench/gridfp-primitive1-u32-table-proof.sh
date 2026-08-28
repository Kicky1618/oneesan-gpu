#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

CXX="${CXX:-g++}"
command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_primitive1_u32_table_proof.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_primitive1_u32_table_proof}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'gridfp-primitive1-u32-table-proof OK' <<<"$out"
grep -Fq 'sectors=14 occupied_min=1 occupied_max=27' <<<"$out"
grep -Fq 'table_entries=14 table_bytes=56 max_value=2674440' <<<"$out"
grep -Fq 'uint32_exact=1 exact=1' <<<"$out"
echo 'gridfp-primitive1-u32-table-proof OK exact=1' >&2
