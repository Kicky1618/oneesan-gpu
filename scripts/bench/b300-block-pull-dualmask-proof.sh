#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"
command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/b300_block_pull_dualmask_proof.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/b300_block_pull_dualmask_proof}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'b300-block-pull-dualmask-proof OK' <<<"$out"
grep -Fq 'basis28=1 symbol_truth_nrlx=1' <<<"$out"
grep -Fq 'endpoint_union_exact=1 exact=1' <<<"$out"
echo 'b300-block-pull-dualmask-proof OK exact=1' >&2
