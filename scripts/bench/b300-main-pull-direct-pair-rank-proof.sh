#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"
command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/b300_main_pull_direct_pair_rank_proof.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/b300_main_pull_direct_pair_rank_proof}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'b300-main-pull-direct-pair-rank-proof OK' <<<"$out"
grep -Fq 'rank_slice_calls=0' <<<"$out"
grep -Fq 'exact=1' <<<"$out"
echo 'b300-main-pull-direct-pair-rank-proof OK exact=1' >&2
