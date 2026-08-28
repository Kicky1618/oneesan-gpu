#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

MODE="${MODE:-dense}"
ARCH="${ARCH:-native}"
PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"
RUNTIME_CACHE_EDGES="${RUNTIME_CACHE_EDGES:-1}"

if [[ "$RUNTIME_CACHE_EDGES" != 0 && "$RUNTIME_CACHE_EDGES" != 1 ]]; then
  echo "RUNTIME_CACHE_EDGES must be 0 or 1" >&2
  exit 2
fi

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
  grouped)
    SRC_REL="src/cuda/gridfp/gridfp_reduced_production_grouped_tile_microprobe.cu"
    ;;
  inplace)
    SRC_REL="src/cuda/gridfp/gridfp_reduced_production_grouped_inplace_microprobe.cu"
    ;;
  cycle)
    SRC_REL="src/cuda/gridfp/gridfp_reduced_production_equal_tile_cycle_microprobe.cu"
    ;;
  shift-cycle)
    SRC_REL="src/cuda/gridfp/gridfp_reduced_production_shift_cycle_microprobe.cu"
    ;;
  turn)
    SRC_REL="src/cuda/gridfp/gridfp_reduced_production_turn_microprobe.cu"
    ;;
  owner-component)
    SRC_REL="src/cuda/gridfp/gridfp_reduced_production_owner_component_microprobe.cu"
    ;;
  owner-lean)
    SRC_REL="src/cuda/gridfp/gridfp_reduced_production_owner_component_lean_microprobe.cu"
    ;;
  owner-subwarp)
    SRC_REL="src/cuda/gridfp/gridfp_reduced_production_owner_subwarp_microprobe.cu"
    ;;
  turn-owner-subwarp)
    SRC_REL="src/cuda/gridfp/gridfp_reduced_production_turn_owner_subwarp_microprobe.cu"
    ;;
  turn-high-owner-subwarp)
    SRC_REL="src/cuda/gridfp/gridfp_reduced_production_turn_high_owner_subwarp_microprobe.cu"
    ;;
  p2p-cycle)
    SRC_REL="src/cuda/gridfp/gridfp_reduced_production_p2p_cycle_microprobe.cu"
    ;;
  two-row-multigpu)
    SRC_REL="src/cuda/gridfp/gridfp_reduced_production_two_row_multigpu_microprobe.cu"
    ;;
  two-row-runtime-multigpu)
    SRC_REL="src/cuda/gridfp/gridfp_reduced_production_two_row_runtime_multigpu_microprobe.cu"
    ;;
  *)
    echo "invalid MODE=$MODE (forward|reverse|register|persistent|dense|edge|grouped|inplace|cycle|shift-cycle|turn|owner-component|owner-lean|owner-subwarp|turn-owner-subwarp|turn-high-owner-subwarp|p2p-cycle|two-row-multigpu|two-row-runtime-multigpu)" >&2
    exit 2
    ;;
esac

SRC="$(repo_path "$SRC_REL")"
DEFAULT_OUT="gridfp_reduced_component_${MODE}"
if [[ "$MODE" == two-row-runtime-multigpu && "$RUNTIME_CACHE_EDGES" == 0 ]]; then
  DEFAULT_OUT="${DEFAULT_OUT}_edgecache0"
fi
OUT="$(build_path "${OUT:-$DEFAULT_OUT}")"
PTXAS_FLAGS=()
if [[ "$PTXAS_VERBOSE" == 1 ]]; then PTXAS_FLAGS+=("-Xptxas=-v"); fi

TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" \
  "${PTXAS_FLAGS[@]}" \
  -DRP_RUNTIME_CACHE_EDGES="$RUNTIME_CACHE_EDGES" \
  "$SRC" -o "$OUT"

echo "built $OUT (mode=$MODE arch=$ARCH runtime_cache_edges=$RUNTIME_CACHE_EDGES)"
