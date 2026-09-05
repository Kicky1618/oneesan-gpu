#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

CXX="${CXX:-c++}"
command -v "$CXX" >/dev/null || { echo "C++ compiler is required" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_support_unrank_len13_table_proof.cpp"
BIN="$ONEESAN_BUILD_DIR/gridfp_support_unrank_len13_table_proof"
mkdir -p "$ONEESAN_BUILD_DIR"
"$CXX" -O3 -std=c++17 -Wall -Wextra -pedantic "$SRC" -o "$BIN"
out="$($BIN)"
echo "$out"
grep -Fq 'gridfp-support-unrank-len13-table-proof OK' <<<"$out"
grep -Fq 'ranks=8192' <<<"$out"
grep -Fq 'unique_masks=8192' <<<"$out"
grep -Fq 'table_bytes=16384' <<<"$out"
grep -Fq 'max_linear_iterations=13' <<<"$out"
grep -Fq 'rank_roundtrip_exact=1' <<<"$out"
grep -Fq 'coverage_all_masks=1' <<<"$out"
echo 'gridfp-support-unrank-len13-table-proof runner OK' >&2
