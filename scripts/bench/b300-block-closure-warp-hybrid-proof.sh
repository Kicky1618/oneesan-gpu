#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"
command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/b300_block_closure_warp_hybrid_proof.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/b300_block_closure_warp_hybrid_proof}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'b300-block-closure-warp-hybrid-proof OK' <<<"$out"
grep -Fq 'disjoint_partition=1 scalar_sum_exact=1 warp_sum_exact=1 rl_filter=valid_source_only exact=1' <<<"$out"
grep -Eq 't[1-9][0-9]*_scalar=[1-9][0-9]*' <<<"$out"
grep -Eq 't[1-9][0-9]*_warp=[1-9][0-9]*' <<<"$out"
echo 'b300-block-closure-warp-hybrid-proof OK exact=1' >&2
