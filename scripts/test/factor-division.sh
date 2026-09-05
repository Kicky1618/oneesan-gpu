#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$(mktemp -d "${TMPDIR:-/tmp}/oneesan-factor-test.XXXXXX")"
echo "Test artifacts: $OUT"
"${CXX:-g++}" -O2 -std=c++17 -Wall -Wextra -fsanitize=undefined \
  "$ROOT/tests/invariant_division_test.cpp" -o "$OUT/division"
"$OUT/division"
if command -v nvcc >/dev/null 2>&1; then
  nvcc -O3 -std=c++17 -arch="${ARCH:-sm_80}" \
    -DTARGET_W=10 -DLOW_LUT_K=5 -DHIGH_LUT_K=4 \
    "$ROOT/tests/factor_division_setup_test.cu" -o "$OUT/setup" \
    >"$OUT/setup.build.log" 2>&1
  "$OUT/setup"
else
  echo 'Host factor setup test: SKIP (nvcc unavailable)'
fi
