#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"
command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/b300_main_pull_operator_proof.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/b300_main_pull_operator_proof}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'b300-main-pull-operator-proof OK' <<<"$out"
grep -Fq 'p_scope=2..Wm1' <<<"$out"
grep -Fq 'identity_copy_required=0' <<<"$out"
grep -Fq 'blocked_to_main_scatter_required=0' <<<"$out"
grep -Fq 'main_atomic_updates_required=0' <<<"$out"
grep -Fq 'pull_terms_max=3 exact=1' <<<"$out"
echo 'b300-main-pull-operator-proof OK exact=1' >&2
