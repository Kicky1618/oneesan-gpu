#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

CXX="${CXX:-g++}"
command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_runtime_owner_u32limb_proof.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_runtime_owner_u32limb_proof}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'gridfp-runtime-owner-u32limb-proof OK' <<<"$out"
grep -Fq 'W_min=8 W_max=28 W_step=2 ngpu_min=2 ngpu_max=8' <<<"$out"
grep -Fq 'groups=16376 owner_cases=114632 previous_shift_failures=37' <<<"$out"
grep -Fq 'max_magic=9757 max_magic_bits=14 max_scale=78056 max_scale_bits=17' <<<"$out"
grep -Fq 'max_shift=52 max_midpoint_hi=110 max_midpoint_bits=39 max_upper=8372291 max_upper_bits=23' <<<"$out"
grep -Fq 'table_entries=11 table_bytes=44 midpoint_mul64=0 mul_wide_u32=0 pure_u32=1 clamp_required=0 exact=1' <<<"$out"
echo 'gridfp-runtime-owner-u32limb-proof OK exact=1 pure_u32=1' >&2
