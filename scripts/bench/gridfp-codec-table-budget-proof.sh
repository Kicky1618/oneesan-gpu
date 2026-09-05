#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

CXX="${CXX:-g++}"
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_codec_table_budget_proof.cpp"
OUT="${OUT:-$ONEESAN_BUILD_DIR/gridfp_codec_table_budget_proof}"
mkdir -p "$(dirname "$OUT")"
"$CXX" -O2 -std=c++17 -Wall -Wextra -pedantic "$SRC" -o "$OUT"
line="$($OUT)"
echo "$line"
grep -Fq 'gridfp-codec-table-budget-proof OK' <<<"$line"
grep -Fq 'choose_max=40116600' <<<"$line"
grep -Fq 'primitive_nonzero=225 primitive_max=8947575' <<<"$line"
grep -Fq 'motzkin_nonzero=435 motzkin_max=569371325796' <<<"$line"
grep -Fq 'old_total_bytes=20648' <<<"$line"
grep -Fq 'max_compact_bytes=5280 max_compact_saved_bytes=15368' <<<"$line"
grep -Fq 'tri_compact_bytes=6120 tri_compact_saved_bytes=14528' <<<"$line"
grep -Fq 'low_arith_bytes=10324 low_arith_saved_bytes=10324' <<<"$line"
grep -Fq 'choose_full_u32_bytes=3364 primitive_full_u32_bytes=3480 motzkin_tri_bytes=3480' <<<"$line"
grep -Fq 'choose_u32=1 primitive_u32=1 motzkin_u32=0 exact=1' <<<"$line"
echo "gridfp-codec-table-budget-proof.sh OK" >&2
