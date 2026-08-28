#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
CXX="${CXX:-g++}"
MAX_W="${MAX_W:-10}"
OUTDIR="${OUTDIR:-$ROOT/build/gridfp-production-signed-shear}"
mkdir -p "$OUTDIR"

SRC="$ROOT/src/cpp/probes/gridfp_reduced_production_signed_shear_probe.cpp"
BIN="$OUTDIR/gridfp_reduced_production_signed_shear_probe"

"$CXX" -O3 -std=c++17 -Wall -Wextra -pedantic "$SRC" -o "$BIN"
"$BIN" "$MAX_W"
