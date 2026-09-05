#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"
command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/b300_block_closure_rank_incremental_proof.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/b300_block_closure_rank_incremental_proof}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'b300-block-closure-rank-incremental-proof OK' <<<"$out"
grep -Fq 'width_max=10' <<<"$out"
grep -Eq 'left_candidates=[1-9][0-9]* right_candidates=[1-9][0-9]*' <<<"$out"
grep -Fq 'repeated_rank_same_scans=0 incremental_rank_delta=1 grouped_masks=1 exact=1' <<<"$out"
echo 'b300-block-closure-rank-incremental-proof OK exact=1' >&2
