#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"
command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/b300_ilp4_partition_proof.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/b300_ilp4_partition_proof}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'b300-ilp4-partition-proof OK' <<<"$out"
grep -Fq 'pattern=base_tid_plus_lane_grid lanes=4 stride=4grid duplicate=0 missing=0 exact=1' <<<"$out"
launch_out="$(python3 "$ONEESAN_ROOT/scripts/bench/b300-main-pull-ilp-production-launch-proof.py")"
printf '%s\n' "$launch_out"
grep -Fq 'b300-main-pull-ilp-production-launch-proof OK scaled_launch=1 exact=1' <<<"$launch_out"
grep -Fq 'lanes=4 ' <<<"$launch_out"
echo 'b300-ilp4-partition-proof OK exact=1 production_launch_scaled=1' >&2
