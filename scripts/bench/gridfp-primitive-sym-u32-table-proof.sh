#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

CXX="${CXX:-g++}"
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_primitive_sym_u32_table_proof.cpp"
OUT="${OUT:-$ONEESAN_BUILD_DIR/gridfp_primitive_sym_u32_table_proof}"
mkdir -p "$(dirname "$OUT")"
"$CXX" -O2 -std=c++17 -Wall -Wextra -pedantic "$SRC" -o "$OUT"
line="$($OUT)"
echo "$line"
grep -Fq 'gridfp-primitive-sym-u32-table-proof OK' <<<"$line"
grep -Fq 'table_cases=870' <<<"$line"
grep -Fq 'nonzero_cells=225' <<<"$line"
grep -Fq 'compact_entries=225' <<<"$line"
grep -Fq 'old_bytes=6960' <<<"$line"
grep -Fq 'compact_bytes=900' <<<"$line"
grep -Fq 'saved_bytes=6060' <<<"$line"
grep -Fq 'max_value=8947575' <<<"$line"
grep -Fq ' exact=1' <<<"$line"
echo "gridfp-primitive-sym-u32-table-proof.sh OK" >&2
