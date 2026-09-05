#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"
command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/b300_rank_delta_window_free_proof.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/b300_rank_delta_window_free_proof}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'b300-rank-delta-window-free-proof OK' <<<"$out"
grep -Fq 'width_max=28' <<<"$out"
grep -Fq 'main_p_free=1 block_pm1_free=1 allowed_checks_required=0 exact=1' <<<"$out"
echo 'b300-rank-delta-window-free-proof OK exact=1' >&2
