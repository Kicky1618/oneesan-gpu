#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

CXX="${CXX:-g++}"
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_depthcode_warpstriped_schedule_selftest.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_depthcode_warpstriped_schedule_selftest}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O2 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'gridfp-depthcode-warpstriped-schedule-selftest OK' <<<"$out"
grep -Fq 'full_warp_required=1' <<<"$out"
grep -Fq 'exact_orbit_column_cover=1' <<<"$out"
grep -Fq 'no_duplicate_work_items=1' <<<"$out"
echo "pattern10-depthcode-warpstriped-schedule-selftest OK" >&2
