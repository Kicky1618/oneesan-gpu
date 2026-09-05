#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"; command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_runtime_blocked_rank_direct_proof.cpp"; BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_runtime_blocked_rank_direct_proof}"; mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"; out="$($BIN)"; printf '%s\n' "$out"
grep -Fq 'gridfp-runtime-blocked-rank-direct-proof OK' <<<"$out"
grep -Fq 'random_cases=1000000 production_W_max=28' <<<"$out"
grep -Fq 'support_insert_zero_exact=1 occupied_sequence_exact=1 fused_block_rank_exact=1' <<<"$out"
echo 'gridfp-runtime-blocked-rank-direct-proof OK exact=1' >&2
