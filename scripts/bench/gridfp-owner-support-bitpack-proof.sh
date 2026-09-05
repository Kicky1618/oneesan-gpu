#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"; command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_owner_support_bitpack_proof.cpp"; BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_owner_support_bitpack_proof}"; mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"; out="$($BIN)"; printf '%s\n' "$out"
grep -Fq 'gridfp-owner-support-bitpack-proof OK' <<<"$out"
grep -Fq 'random_cases=1000000 production_W_max=28' <<<"$out"
grep -Fq 'outer_expand_exact=1 local_expand_exact=1 label_reverse_exact=1' <<<"$out"
echo 'gridfp-owner-support-bitpack-proof OK exact=1' >&2
