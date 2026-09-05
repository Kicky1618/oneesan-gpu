#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"
command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/b300_block_closure_warp_parallel_prefix_proof.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/b300_block_closure_warp_parallel_prefix_proof}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'b300-block-closure-warp-parallel-prefix-proof OK' <<<"$out"
grep -Fq 'serial_scan_equivalent=1' <<<"$out"
grep -Fq 'candidate_barrier_ballot_equivalent=1' <<<"$out"
grep -Fq 'rank_delta_prefix_scan_equivalent=1 exact=1' <<<"$out"
echo 'b300-block-closure-warp-parallel-prefix-proof OK exact=1' >&2
