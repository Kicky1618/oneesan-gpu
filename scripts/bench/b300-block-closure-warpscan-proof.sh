#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"
command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/b300_block_closure_warpscan_proof.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/b300_block_closure_warpscan_proof}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'b300-block-closure-warpscan-proof OK' <<<"$out"
grep -Fq 'exhaustive_width_max=12' <<<"$out"
grep -Fq 'closure_states=54418' <<<"$out"
grep -Fq 'prefix_balance=exact prefix_min_break=exact' <<<"$out"
grep -Fq 'rank_delta_prefix_scan=exact shared_rank_queue_required=0 exact=1' <<<"$out"
echo 'b300-block-closure-warpscan-proof OK exhaustive=54418 exact=1' >&2
