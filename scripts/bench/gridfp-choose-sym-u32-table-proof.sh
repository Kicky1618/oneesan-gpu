#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

CXX="${CXX:-g++}"
command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_choose_sym_u32_table_proof.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_choose_sym_u32_table_proof}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'gridfp-choose-sym-u32-table-proof OK' <<<"$out"
grep -Fq 'production_n_max=28 valid_choose_cases=435 sym_entries=225 tri_entries=435 full_entries=841' <<<"$out"
grep -Fq 'old_bytes=6728 sym_bytes=900 tri_bytes=1740 full_bytes=3364' <<<"$out"
grep -Fq 'sym_saved_bytes=5828 tri_saved_bytes=4988 full_saved_bytes=3364 max_value=40116600' <<<"$out"
grep -Fq 'symmetry_exact=1 triangle_exact=1 full_shape_exact=1 row_base_exact=1 uint32_exact=1 exact=1' <<<"$out"
echo 'gridfp-choose-sym-u32-table-proof OK exact=1' >&2
