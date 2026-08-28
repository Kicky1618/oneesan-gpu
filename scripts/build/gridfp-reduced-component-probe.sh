#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

MODE="${MODE:-dense}"
ARCH="${ARCH:-native}"
PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"

case "$MODE" in
  forward)
    SRC_REL="src/cuda/gridfp/gridfp_reduced_production_component_microprobe.cu"
    ;;
  reverse)
    SRC_REL="src/cuda/gridfp/gridfp_reduced_production_component_reverse_microprobe.cu"
    ;;
  register)
    SRC_REL="src/cuda/gridfp/gridfp_reduced_production_component_register_microprobe.cu"
    ;;
  persistent)
    SRC_REL="src/cuda/gridfp/gridfp_reduced_production_component_persistent_microprobe.cu"
    ;;
  dense)
    SRC_REL="src/cuda/gridfp/gridfp_reduced_production_component_dense_microprobe.cu"
    ;;
  edge)
    SRC_REL="src/cuda/gridfp/gridfp_reduced_production_row_edge_microprobe.cu"
    ;;
  *)
    echo "invalid MODE=$MODE (forward|reverse|register|persistent|dense|edge)" >&2
    exit 2
    ;;
esac

SRC="$(repo_path "$SRC_REL")"
OUT="$(build_path "${OUT:-gridfp_reduced_component_${MODE}}")"
PTXAS_FLAGS=()
if [[ "$PTXAS_VERBOSE" == 1 ]]; then PTXAS_FLAGS+=("-Xptxas=-v"); fi

TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" \
  "${PTXAS_FLAGS[@]}" "$SRC" -o "$OUT"

echo "built $OUT (mode=$MODE arch=$ARCH)"
