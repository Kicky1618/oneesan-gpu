#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

CXX="${CXX:-g++}"
command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_runtime_owner_w28_ngpu8_direct_proof.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_runtime_owner_w28_ngpu8_direct_proof}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'gridfp-runtime-owner-w28-ngpu8-direct-proof OK W=28 ngpu=8 groups=8192' <<<"$out"
grep -Fq 'magic=9513 shift=17 max_owner=7' <<<"$out"
grep -Fq 'meta_loads=0 variable_shift=0 ngpu_mul=0 scale_mul=0 product_lo=0 mul64=0 clamp=0 exact=1' <<<"$out"
echo 'gridfp-runtime-owner-w28-ngpu8-direct-proof OK exact=1 direct=1' >&2
