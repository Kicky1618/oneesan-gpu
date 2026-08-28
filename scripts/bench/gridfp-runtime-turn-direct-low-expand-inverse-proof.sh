#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"; command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_runtime_turn_direct_low_expand_inverse_proof.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_runtime_turn_direct_low_expand_inverse_proof}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"; printf '%s\n' "$out"
grep -Fq 'gridfp-runtime-turn-direct-low-expand-inverse-proof OK' <<<"$out"
grep -Fq 'exhaustive_width_max=12' <<<"$out"
grep -Fq 'random_width=28 random_cases=1000000' <<<"$out"
grep -Fq 'source_scope=main_only' <<<"$out"
grep -Fq 'reverse_basis=mirrored_Q_Wm2' <<<"$out"
grep -Fq 'direct_reverse_validity=low_to_high' <<<"$out"
grep -Fq 'destination_mirror_passes=0 source_mirror_passes=0' <<<"$out"
grep -Fq 'inverse_set_exact=1' <<<"$out"
echo 'gridfp-runtime-turn-direct-low-expand-inverse-proof OK exact=1' >&2
