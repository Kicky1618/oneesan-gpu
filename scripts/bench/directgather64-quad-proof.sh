#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

CXX="${CXX:-g++}"
OUT="${OUT:-$ONEESAN_BUILD_DIR/gridfp_directgather64_quad_proof}"
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_directgather64_quad_proof.cpp"
command -v "$CXX" >/dev/null || { echo "$CXX required" >&2; exit 2; }
"$CXX" -O3 -std=c++17 -Wall -Wextra "$SRC" -o "$OUT"
result="$("$OUT")"
printf '%s\n' "$result"
for token in \
  'gridfp-directgather64-quad-proof OK' \
  'scheduler_exact=1' \
  'tail_1_2_3_exact=1' \
  'dense64_decode_exact=1' \
  'sparse64_decode_exact=1' \
  'quad4_sum_exact=1'; do
  grep -Fq "$token" <<<"$result" || { echo "missing proof token: $token" >&2; exit 3; }
done
echo "directgather64-quad-proof OK" >&2
