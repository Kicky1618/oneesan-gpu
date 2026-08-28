#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

CXX="${CXX:-g++}"
if ! command -v "$CXX" >/dev/null; then
  echo "$CXX not found" >&2
  exit 2
fi

SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_runtime_fastdiv64_proof.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_runtime_fastdiv64_proof}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"

grep -Fq 'gridfp-runtime-fastdiv64-proof OK' <<<"$out"
grep -Fq 'small_bits=12' <<<"$out"
grep -Fq 'random64=2000000' <<<"$out"
grep -Fq 'production_max_numerator=473397057701' <<<"$out"
grep -Fq 'quotient_error_bound=1 product_overflow_checked=1 exact=1' <<<"$out"

echo 'gridfp-runtime-fastdiv64-proof OK exact reciprocal divmod64' >&2
