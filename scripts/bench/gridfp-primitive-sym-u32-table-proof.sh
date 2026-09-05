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
grep -Fq 'compact_entries=225 full_entries=870' <<<"$line"
grep -Fq 'old_bytes=6960 compact_bytes=900 full_bytes=3480' <<<"$line"
grep -Fq 'compact_saved_bytes=6060 full_saved_bytes=3480' <<<"$line"
grep -Fq 'max_value=8947575' <<<"$line"
grep -Fq 'parity_sparse_exact=1 full_shape_exact=1 row_base_exact=1 uint32_exact=1 exact=1' <<<"$line"
echo "gridfp-primitive-sym-u32-table-proof.sh OK" >&2
