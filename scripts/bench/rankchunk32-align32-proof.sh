#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"
if ! command -v "$CXX" >/dev/null; then echo "$CXX not found" >&2; exit 2; fi
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_rankchunk32_align32.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_rankchunk32_align32}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'gridfp-rankchunk32-align32 OK' <<<"$out"
grep -Fq 'block=32 height_align=32' <<<"$out"
grep -Fq 'max_block_base_loads_per_warp=1' <<<"$out"
grep -Fq 'max_padding_per_height=31' <<<"$out"
grep -Fq 'max_padding_entries_per_owner=930 max_padding_bytes_per_owner=3720' <<<"$out"
grep -Fq 'direct_block_exact=1' <<<"$out"
echo 'rankchunk32-align32-proof OK block=32 max_loads=1 max_padding_bytes_per_owner=3720' >&2
