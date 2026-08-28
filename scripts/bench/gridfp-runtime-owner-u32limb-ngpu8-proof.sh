#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

CXX="${CXX:-g++}"
command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_runtime_owner_u32limb_ngpu8_proof.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_runtime_owner_u32limb_ngpu8_proof}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'gridfp-runtime-owner-u32limb-ngpu8-proof OK' <<<"$out"
grep -Fq 'W_min=8 W_max=28 W_step=2 ngpu=8 groups=16376 shift_bias=3' <<<"$out"
grep -Fq 'min_effective_shift=16 max_effective_shift=49 max_magic=9757 max_magic_bits=14 max_midpoint_hi=110' <<<"$out"
grep -Fq 'ngpu_mul=0 scale_mul=0 midpoint_mul64=0 pure_u32=1 exact=1' <<<"$out"
echo 'gridfp-runtime-owner-u32limb-ngpu8-proof OK exact=1 pure_u32=1 ngpu_mul=0' >&2
