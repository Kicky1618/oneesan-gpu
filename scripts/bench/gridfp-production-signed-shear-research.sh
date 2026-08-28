#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
CXX="${CXX:-g++}"
MAX_W="${MAX_W:-10}"
OUTDIR="${OUTDIR:-$ROOT/build/gridfp-production-signed-shear}"
mkdir -p "$OUTDIR"

build_and_run() {
  local stem="$1"
  local src="$ROOT/src/cpp/probes/${stem}.cpp"
  local bin="$OUTDIR/$stem"
  echo "=== $stem ===" >&2
  "$CXX" -O3 -std=c++17 -Wall -Wextra -pedantic "$src" -o "$bin"
  "$bin" "$MAX_W"
}

build_and_run gridfp_reduced_production_signed_shear_probe
build_and_run gridfp_reduced_production_nilpotent_probe
build_and_run gridfp_reduced_production_component_alignment_probe
