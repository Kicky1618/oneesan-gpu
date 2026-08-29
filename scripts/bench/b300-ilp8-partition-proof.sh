#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"
command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/b300_ilp8_partition_proof.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/b300_ilp8_partition_proof}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'b300-ilp8-partition-proof OK' <<<"$out"
grep -Fq 'destinations_per_thread=8 exact_partition=1' <<<"$out"
grep -Fq 'launch_blocks=ceil_n_over_8threads_capped65535' <<<"$out"
echo 'b300-ilp8-partition-proof OK exact=1' >&2
