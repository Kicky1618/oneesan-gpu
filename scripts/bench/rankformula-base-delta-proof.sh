#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

CXX="${CXX:-g++}"
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_rankformula_base_delta.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_rankformula_base_delta}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'gridfp-rankformula-base-delta OK' <<<"$out"
grep -Fq 'codes=1201917' <<<"$out"
grep -Fq 'transitions=3720805' <<<"$out"
grep -Fq 'mismatches=0' <<<"$out"
grep -Fq 'min_base_delta=-12969' <<<"$out"
grep -Fq 'max_base_delta=14873' <<<"$out"
grep -Fq 'int16_exact=1' <<<"$out"
grep -Fq 'base_values_per_lookup=1' <<<"$out"
echo 'rankformula-base-delta-proof OK exact_w28=1 int16=1 base_values_per_lookup=1' >&2
