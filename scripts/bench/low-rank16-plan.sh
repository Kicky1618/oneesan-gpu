#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"
if ! command -v "$CXX" >/dev/null; then echo "$CXX not found" >&2; exit 2; fi
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_low_rank16_plan.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_low_rank16_plan}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'gridfp-low-rank16-plan OK' <<<"$out"
grep -Fq 'W=28 low_k=14 high_k=13' <<<"$out"
grep -Fq 'owner_invariant=1' <<<"$out"
grep -Fq 'dense_bytes_per_code=28' <<<"$out"
grep -Fq 'sparse_model=mask16+offset32+rank16' <<<"$out"
for g in 0 1 2 3 4 5 6 7; do grep -Fq "owner=$g " <<<"$out"; done
echo 'low-rank16-plan OK W=28 owner_invariant=1' >&2
