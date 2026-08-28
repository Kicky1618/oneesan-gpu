#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"; command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_runtime_turn_local_sector_table_proof.cpp"; BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_runtime_turn_local_sector_table_proof}"; mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"; out="$($BIN)"; printf '%s\n' "$out"
grep -Fq 'gridfp-runtime-turn-local-sector-table-proof OK' <<<"$out"
grep -Fq 'entries=550 bytes=2200 max_end=510468519' <<<"$out"
grep -Fq 'random_cases=1000000 production_W_max=28 embedded_exact=1 parity_compact=1 binary_exact=1 max_binary_comparisons=4' <<<"$out"
echo 'gridfp-runtime-turn-local-sector-table-proof OK exact=1' >&2
