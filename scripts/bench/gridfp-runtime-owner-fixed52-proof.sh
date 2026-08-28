#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

CXX="${CXX:-g++}"
command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }

SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_runtime_owner_fixed52_proof.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_runtime_owner_fixed52_proof}"
mkdir -p "$(dirname "$BIN")"

"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"

grep -Fq 'gridfp-runtime-owner-fixed52-proof OK' <<<"$out"
grep -Fq 'W_min=8 W_max=28 W_step=2 ngpu_min=2 ngpu_max=8' <<<"$out"
grep -Fq 'groups=16376 owner_cases=114632 shift=52 shift51_failures=3' <<<"$out"
grep -Fq 'product_bits=55 table_entries=11 table_bytes=88 owner_lt_ngpu=1 clamp_required=0 mulhi=0 correction=0 exact=1' <<<"$out"

echo 'gridfp-runtime-owner-fixed52-proof OK exact=1' >&2
