#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

CXX="${CXX:-g++}"
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_cross5_rankmask_shape.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_cross5_rankmask_shape}"
mkdir -p "$(dirname "$BIN")"

"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'cases=6075' <<<"$out"
grep -Fq 'mask_set_exact=1 upper_bits_zero=1 direct3_exact=1 max_popcount=3' <<<"$out"
grep -Fq 'popcount_hist=5855,187,32,1,0,0' <<<"$out"
grep -Fq 'allowed_masks=0,1,2,3,5,7' <<<"$out"
echo 'cross5-rankmask-shape-proof OK direct3=exact upper_bits=unreachable' >&2
