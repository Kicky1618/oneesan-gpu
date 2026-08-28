#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"; command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_runtime_outer_group_table_proof.cpp"; BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_runtime_outer_group_table_proof}"; mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"; out="$($BIN)"; printf '%s\n' "$out"
grep -Fq 'gridfp-runtime-outer-group-table-proof OK' <<<"$out"
grep -Fq 'W_configs=11 entries=99 group_bytes=396 prefix_bytes=792 total_bytes=1188' <<<"$out"
grep -Fq 'max_group=1805186805 max_prefix=471591870896 row_bases_exact=1 embedded_exact=1' <<<"$out"
echo 'gridfp-runtime-outer-group-table-proof OK embedded_exact=1' >&2
