#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"
if ! command -v "$CXX" >/dev/null; then echo "$CXX not found" >&2; exit 2; fi
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_rankstream32_warpbase.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_rankstream32_warpbase}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'gridfp-rankstream32-warpbase OK' <<<"$out"
grep -Fq 'cases=1024' <<<"$out"
grep -Fq 'max_block_base_loads_per_warp=2' <<<"$out"
grep -Fq 'lane0_source_always_active=1' <<<"$out"
grep -Fq 'second_source_active_if_needed=1' <<<"$out"
grep -Fq 'direct_block_exact=1' <<<"$out"
echo 'rankstream32-warpbase-proof OK cases=1024 max_block_base_loads_per_warp=2' >&2
