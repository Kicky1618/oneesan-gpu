#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

CXX="${CXX:-g++}"
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_rankformula_plan.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_rankformula_plan}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'gridfp-rankformula-plan OK' <<<"$out"
grep -Fq 'codes=1201917' <<<"$out"
grep -Fq 'transitions=3720805' <<<"$out"
grep -Fq 'mismatches=0' <<<"$out"
grep -Fq 'dense_base_bytes_per_owner=983040' <<<"$out"
grep -Fq 'rankstream_bytes=0' <<<"$out"
grep -Fq 'source_height_delta=2' <<<"$out"
echo 'rankformula-plan OK exact_w28=1 streamless=1 dense_base_960kib_per_owner=1' >&2
