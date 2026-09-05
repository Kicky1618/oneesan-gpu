#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

CXX="${CXX:-g++}"
if ! command -v "$CXX" >/dev/null; then
  echo "$CXX not found" >&2
  exit 2
fi
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_cross5_rankmask.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_cross5_rankmask}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'gridfp-cross5-rankmask OK' <<<"$out"
grep -Fq 'states=26 keys=243' <<<"$out"
grep -Fq 'rankmask_bytes=6318 meta_bytes=243 constant_loads_per_chunk=2' <<<"$out"
grep -Fq 'ordinal_popcount_runtime=0 meta_lcount_exact=1 meta_delta_exact=1 candidate_set_exact=1 state_exact=1 halt_exact=1' <<<"$out"
echo 'cross5-rankmask-proof OK rankmask_bytes=6318 meta_bytes=243 constant_loads_per_chunk=2 ordinal_popcount_runtime=0' >&2
