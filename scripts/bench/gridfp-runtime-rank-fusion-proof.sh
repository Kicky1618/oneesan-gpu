#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"; command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_runtime_rank_fusion_proof.cpp"; BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_runtime_rank_fusion_proof}"; mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"; out="$($BIN)"; printf '%s\n' "$out"
grep -Fq 'gridfp-runtime-rank-fusion-proof OK' <<<"$out"
grep -Fq 'random_cases=1000000 production_W_max=28' <<<"$out"
grep -Fq 'main_support_rank_exact=1 blocked_support_rank_exact=1 one_high_to_low_support_pass=1' <<<"$out"
echo 'gridfp-runtime-rank-fusion-proof OK exact=1' >&2
