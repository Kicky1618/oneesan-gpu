#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"
command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/b300_ilp2_partition_proof.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/b300_ilp2_partition_proof}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'b300-ilp2-partition-proof OK' <<<"$out"
grep -Fq 'pattern=base_tid_plus_grid stride=2grid duplicate=0 missing=0 exact=1' <<<"$out"
echo 'b300-ilp2-partition-proof OK exact=1' >&2
