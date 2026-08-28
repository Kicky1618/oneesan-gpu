#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
CXX="${CXX:-g++}"
MAX_W="${MAX_W:-12}"
OUT="${OUT:-$ROOT/build/two_cell_packed_local_shear_probe}"
mkdir -p "$(dirname "$OUT")"

"$CXX" -O3 -std=c++17 -Wall -Wextra -pedantic \
  "$ROOT/src/cpp/probes/two_cell_packed_local_shear_probe.cpp" \
  -o "$OUT"

"$OUT" "$MAX_W"
