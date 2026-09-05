#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"
command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/b300_rank_delta_recurrence_proof.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/b300_rank_delta_recurrence_proof}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'b300-rank-delta-recurrence-proof OK' <<<"$out"
grep -Fq 'signed_delta=1 step_O1=1 exact=1' <<<"$out"
echo 'b300-rank-delta-recurrence-proof OK exact=1' >&2
