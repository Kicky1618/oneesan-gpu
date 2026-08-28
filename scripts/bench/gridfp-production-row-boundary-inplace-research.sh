#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
CXX="${CXX:-g++}"
MAX_W="${MAX_W:-11}"
RUN_CUDA="${RUN_CUDA:-0}"
CUDA_W="${CUDA_W:-10}"
BLOCKS="${BLOCKS:-4096}"
MOD="${MOD:-4294967291}"
ARCH="${ARCH:-native}"
OUTDIR="${OUTDIR:-$ROOT/build/gridfp_row_boundary_inplace_research}"
mkdir -p "$OUTDIR"

build_cpp() {
  local stem="$1"
  local src="$ROOT/src/cpp/probes/${stem}.cpp"
  local bin="$OUTDIR/$stem"
  echo "=== $stem ===" >&2
  "$CXX" -O3 -std=c++17 -Wall -Wextra -pedantic "$src" -o "$bin"
  "$bin" "$MAX_W"
}

build_cpp gridfp_reduced_production_row_entry_inplace_probe
build_cpp gridfp_reduced_production_row_edge_inplace_probe

if [[ "$RUN_CUDA" == 0 ]]; then
  exit 0
fi
if [[ "$RUN_CUDA" != 1 ]]; then
  echo "RUN_CUDA must be 0 or 1" >&2
  exit 2
fi

for mode in entry-inplace final-inplace; do
  out="$OUTDIR/$mode"
  MODE="$mode" ARCH="$ARCH" OUT="$out" \
    bash "$ROOT/scripts/build/gridfp-reduced-component-probe.sh"
  "$out" "$CUDA_W" "$BLOCKS" "$MOD"
done
