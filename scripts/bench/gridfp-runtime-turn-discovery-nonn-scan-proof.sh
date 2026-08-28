#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

CXX="${CXX:-g++}"
command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_runtime_turn_discovery_nonn_scan_proof.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_runtime_turn_discovery_nonn_scan_proof}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'gridfp-runtime-turn-discovery-nonn-scan-proof OK' <<<"$out"
grep -Fq 'exhaustive_width_max=10' <<<"$out"
grep -Fq 'random_width_max=28 random_cases=1000000' <<<"$out"
grep -Fq 'candidate_sequence_exact=1 balance_exact=1 stop_exact=1' <<<"$out"
grep -Fq 'nonn_iterations_le_full=1' <<<"$out"
echo 'gridfp-runtime-turn-discovery-nonn-scan-proof OK exact=1' >&2
