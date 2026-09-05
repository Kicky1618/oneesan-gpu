#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

CXX="${CXX:-g++}"
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_rankformula_inline_cross.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_rankformula_inline_cross}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'gridfp-rankformula-inline-cross OK' <<<"$out"
grep -Fq 'states=25 keys=243 cases=6075' <<<"$out"
grep -Fq 'cross_lut_loads=0' <<<"$out"
grep -Fq 'single_symbol_scan=1' <<<"$out"
echo 'rankformula-inline-cross-proof OK states=25 keys=243 cross_lut_loads=0' >&2
