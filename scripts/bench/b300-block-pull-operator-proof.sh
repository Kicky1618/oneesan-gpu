#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"
command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/b300_block_pull_operator_proof.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/b300_block_pull_operator_proof}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'b300-block-pull-operator-proof OK' <<<"$out"
grep -Fq 'exhaustive_width_max=12' <<<"$out"
grep -Fq 'p_scope=2..Wm1 deferred_drop_position=p' <<<"$out"
grep -Fq 'rl_gate=prefix_height_exact' <<<"$out"
grep -Eq 'rl_candidates=[1-9][0-9]* rl_rejected=[1-9][0-9]*' <<<"$out"
grep -Fq 'block_memset_required=0' <<<"$out"
grep -Fq 'block_atomic_updates_required=0' <<<"$out"
grep -Fq 'pull_terms_max=5 exact=1' <<<"$out"
echo 'b300-block-pull-operator-proof OK exact=1 deferred_drop_position=p rl_gate=prefix_height_exact' >&2
