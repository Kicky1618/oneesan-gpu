#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
CXX="${CXX:-g++}"
MAX_W="${MAX_W:-12}"
NGPU="${NGPU:-8}"
OUTDIR="${OUTDIR:-$ROOT/build/gridfp_production_slab_cycle_research}"
mkdir -p "$OUTDIR"

SRC="$ROOT/src/cpp/probes/gridfp_reduced_production_slab_cycle_probe.cpp"
BIN="$OUTDIR/gridfp_reduced_production_slab_cycle_probe"

"$CXX" -O3 -std=c++17 -Wall -Wextra -pedantic "$SRC" -o "$BIN"
"$BIN" "$MAX_W" "$NGPU"
