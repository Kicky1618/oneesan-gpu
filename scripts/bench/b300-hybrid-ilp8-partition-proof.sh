#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

CXX="${CXX:-g++}"
command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/b300_hybrid_ilp8_partition_proof.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/b300_hybrid_ilp8_partition_proof}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'b300-hybrid-ilp8-partition-proof OK' <<<"$out"
grep -Fq 'threshold_rule=n_ge_threshold_uses_ilp8' <<<"$out"
grep -Fq 'ilp4_destinations=4 ilp8_destinations=8' <<<"$out"
grep -Fq 'block_cap=65535 exact_partition=1' <<<"$out"
echo 'b300-hybrid-ilp8-partition-proof OK exact=1' >&2
