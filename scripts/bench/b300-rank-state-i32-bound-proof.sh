#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"
command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/b300_rank_state_i32_bound_proof.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/b300_rank_state_i32_bound_proof}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'b300-rank-state-i32-bound-proof OK' <<<"$out"
grep -Fq 'runtime_guard=max_group_size_le_int32max storage_bytes=8 delta_bits=32 height_bits=8' <<<"$out"
grep -Fq 'pack_roundtrip=1 exact=1' <<<"$out"
echo 'b300-rank-state-i32-bound-proof OK exact=1' >&2
