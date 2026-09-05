#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

CXX="${CXX:-g++}"
command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }

SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_runtime_owner_fixed54_proof.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_runtime_owner_fixed54_proof}"
mkdir -p "$(dirname "$BIN")"

"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"

grep -Fq 'gridfp-runtime-owner-fixed54-proof OK' <<<"$out"
grep -Fq 'W_min=8 W_max=28 W_step=2' <<<"$out"
grep -Fq 'groups=16376 owner_cases=245640 shift=54 shift53_failures=3' <<<"$out"
grep -Fq 'product_bits=58 table_entries=11 table_bytes=88 mulhi=0 correction=0 exact=1' <<<"$out"

echo 'gridfp-runtime-owner-fixed54-proof OK exact=1' >&2
