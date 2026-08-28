#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"; command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_runtime_turn_direct_low_compress_step_proof.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_runtime_turn_direct_low_compress_step_proof}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"; printf '%s\n' "$out"
grep -Fq 'gridfp-runtime-turn-direct-low-compress-step-proof OK' <<<"$out"
grep -Fq 'exhaustive_width_max=12' <<<"$out"
grep -Fq 'random_width=28 random_main=1000000 random_blocked=1000000' <<<"$out"
grep -Fq 'local_cases=NN,NR,RN' <<<"$out"
grep -Fq 'rr_closure=closure_match_right' <<<"$out"
grep -Fq 'generic_include_calls=0' <<<"$out"
grep -Fq 'step_exact=1' <<<"$out"
echo 'gridfp-runtime-turn-direct-low-compress-step-proof OK exact=1' >&2
