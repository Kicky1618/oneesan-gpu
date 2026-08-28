#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"; command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_runtime_owner_reciprocal_proof.cpp"; BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_runtime_owner_reciprocal_proof}"; mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"; out="$($BIN)"; printf '%s\n' "$out"
grep -Fq 'gridfp-runtime-owner-reciprocal-proof OK' <<<"$out"
grep -Fq 'W_min=8 W_max=28 W_step=2' <<<"$out"
grep -Fq 'groups=16376 owner_cases=245640' <<<"$out"
grep -Fq 'table_entries=11 table_bytes=176 quotient_error_bound=1 exact=1' <<<"$out"
echo 'gridfp-runtime-owner-reciprocal-proof OK exact=1' >&2
