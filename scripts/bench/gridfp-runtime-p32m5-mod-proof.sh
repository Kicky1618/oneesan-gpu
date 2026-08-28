#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

CXX="${CXX:-g++}"
if ! command -v "$CXX" >/dev/null; then
  echo "$CXX not found" >&2
  exit 2
fi

SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_runtime_p32m5_mod_proof.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_runtime_p32m5_mod_proof}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"

grep -Fq 'gridfp-runtime-p32m5-mod-proof OK' <<<"$out"
grep -Fq 'modulus=4294967291 max_acc_mag=171798691600 max_hi=39 max_folded=4294967490' <<<"$out"
grep -Fq 'magnitude_cases=196 exact_cases=391' <<<"$out"
grep -Fq 'one_subtraction_bound=1 signed_exact=1 division_free_fast_path=1' <<<"$out"

echo 'gridfp-runtime-p32m5-mod-proof OK modulus=4294967291 exact_cases=391 one_subtraction_bound=1' >&2
