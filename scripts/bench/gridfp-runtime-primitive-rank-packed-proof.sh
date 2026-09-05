#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

CXX="${CXX:-g++}"
command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_runtime_primitive_rank_packed_proof.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_runtime_primitive_rank_packed_proof}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'gridfp-runtime-primitive-rank-packed-proof OK' <<<"$out"
grep -Fq 'primitive_cases=3707851 rank_threshold_cells=91' <<<"$out"
grep -Fq 'shared_packed_entries=104 shared_packed_bytes=416 max_value=742900' <<<"$out"
grep -Fq 'threshold_loads=46762038 exact=1' <<<"$out"
echo 'gridfp-runtime-primitive-rank-packed-proof OK exact=1' >&2
