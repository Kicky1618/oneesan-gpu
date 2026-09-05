#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"; command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_materialize_primitive_setbits_proof.cpp"; BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_materialize_primitive_setbits_proof}"; mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"; out="$($BIN)"; printf '%s\n' "$out"
grep -Fq 'gridfp-materialize-primitive-setbits-proof OK' <<<"$out"
grep -Fq 'random_cases=1000000 production_len_max=28' <<<"$out"
grep -Fq 'output_exact=1 traversal=support_setbits' <<<"$out"
echo 'gridfp-materialize-primitive-setbits-proof OK exact=1' >&2
