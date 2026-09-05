#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"
command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/b300_block_closure_warp_batch_proof.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/b300_block_closure_warp_batch_proof}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'b300-block-closure-warp-batch-proof OK' <<<"$out"
grep -Fq 'exhaustive_width_max=12' <<<"$out"
grep -Eq 'rejected_candidates=[1-9][0-9]*' <<<"$out"
grep -Fq 'warp_lanes=32 rl_filter=valid_source_only one_source_load_per_lane=1 modular_tree_exact=1 exact=1' <<<"$out"
echo 'b300-block-closure-warp-batch-proof OK exact=1 rl_filter=valid_source_only' >&2
