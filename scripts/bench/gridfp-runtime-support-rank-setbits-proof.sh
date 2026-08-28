#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

CXX="${CXX:-g++}"
if ! command -v "$CXX" >/dev/null; then echo "$CXX not found" >&2; exit 2; fi
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_runtime_support_rank_setbits_proof.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_runtime_support_rank_setbits_proof}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'gridfp-runtime-support-rank-setbits-proof OK' <<<"$out"
grep -Fq 'exhaustive_len_max=16' <<<"$out"
grep -Fq 'random_cases=1000000' <<<"$out"
grep -Fq 'runtime_len_max=28 old_scan=len new_scan=ones exact=1' <<<"$out"
echo 'gridfp-runtime-support-rank-setbits-proof OK exact=1' >&2
