#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"
command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/b300_group_rank_drop_insert_proof.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/b300_group_rank_drop_insert_proof}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'b300-group-rank-drop-insert-proof OK' <<<"$out"
grep -Fq 'width_max=10' <<<"$out"
grep -Fq 'fixed_group_masks=1 drop_rank_exact=1 insert_rank_exact=1' <<<"$out"
echo 'b300-group-rank-drop-insert-proof OK exact=1' >&2
