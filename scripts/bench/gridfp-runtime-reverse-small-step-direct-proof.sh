#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"; command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_runtime_reverse_small_step_direct_proof.cpp"; BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_runtime_reverse_small_step_direct_proof}"; mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"; out="$($BIN)"; printf '%s\n' "$out"
grep -Fq 'gridfp-runtime-reverse-small-step-direct-proof OK' <<<"$out"
grep -Fq 'random_cases=1000000 production_W_max=28' <<<"$out"
grep -Fq 'direct_projection_exact=1 direct_step_exact=1' <<<"$out"
echo 'gridfp-runtime-reverse-small-step-direct-proof OK exact=1' >&2
