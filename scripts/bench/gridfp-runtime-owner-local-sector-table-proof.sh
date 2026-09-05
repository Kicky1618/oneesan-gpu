#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"; command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_runtime_owner_local_sector_table_proof.cpp"; BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_runtime_owner_local_sector_table_proof}"; mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"; out="$($BIN)"; printf '%s\n' "$out"
grep -Fq 'gridfp-runtime-owner-local-sector-table-proof OK' <<<"$out"
grep -Fq 'entries=1100 bytes=4400 max_end=448876754' <<<"$out"
grep -Fq 'random_cases=1000000 production_W_max=28 embedded_exact=1 binary_exact=1 max_binary_comparisons=4' <<<"$out"
echo 'gridfp-runtime-owner-local-sector-table-proof OK exact=1' >&2
