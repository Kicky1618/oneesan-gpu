#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"; command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_runtime_support_unrank_early_exit_proof.cpp"; BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_runtime_support_unrank_early_exit_proof}"; mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"; out="$($BIN)"; printf '%s\n' "$out"
grep -Fq 'gridfp-runtime-support-unrank-early-exit-proof OK' <<<"$out"
grep -Fq 'random_cases=1000000 production_len_max=28 exact=1 zero_tail_break=1 forced_one_suffix=1' <<<"$out"
echo 'gridfp-runtime-support-unrank-early-exit-proof OK exact=1' >&2
