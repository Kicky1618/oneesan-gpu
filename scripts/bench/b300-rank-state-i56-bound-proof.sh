#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"
command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/b300_rank_state_i56_bound_proof.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/b300_rank_state_i56_bound_proof}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'b300-rank-state-i56-bound-proof OK' <<<"$out"
grep -Fq 'width_max=28 full_state_bound=385719506620' <<<"$out"
grep -Fq 'signed_delta_bits=56 height_bits=8 storage_bytes=8 fallback_required=0' <<<"$out"
grep -Fq 'exact=1' <<<"$out"
echo 'b300-rank-state-i56-bound-proof OK exact=1 fallback_required=0' >&2
