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
  p2p-cycle)
    SRC_REL="src/cuda/gridfp/gridfp_reduced_production_p2p_cycle_microprobe.cu"
    ;;
  p2p-token-cycle)
    SRC_REL="src/cuda/gridfp/gridfp_reduced_production_p2p_token_cycle_microprobe.cu"
    ;;
  p2p-token-plan)
    SRC_REL="src/cuda/gridfp/gridfp_reduced_production_p2p_token_queue_plan_microprobe.cu"
    ;;
  p2p-pair-queue)
    SRC_REL="src/cuda/gridfp/gridfp_reduced_production_p2p_pair_queue_microprobe.cu"
    ;;
  p2p-traffic)
    SRC_REL="src/cuda/gridfp/gridfp_reduced_production_p2p_traffic_microprobe.cu"
    ;;
  p2p-traffic-matrix)
    SRC_REL="src/cuda/gridfp/gridfp_reduced_production_p2p_traffic_matrix_microprobe.cu"
    ;;
  p2p-capability)
    SRC_REL="src/cuda/gridfp/gridfp_reduced_production_p2p_capability_probe.cu"
    ;;
  p2p-mailbox)
    SRC_REL="src/cuda/gridfp/gridfp_reduced_production_p2p_mailbox_ring_microprobe.cu"
    ;;
  p2p-owner-lut)
    SRC_REL="src/cuda/gridfp/gridfp_reduced_production_p2p_owner_lut_probe.cu"
    ;;
  support-rank)
    SRC_REL="src/cuda/gridfp/gridfp_reduced_production_grouped_support_rank_probe.cu"
    ;;
  *)
    echo "invalid MODE=$MODE (forward|reverse|register|persistent|dense|edge|grouped|inplace|cycle|shift-cycle|turn|owner-component|owner-lean|p2p-cycle|p2p-token-cycle|p2p-token-plan|p2p-pair-queue|p2p-traffic|p2p-traffic-matrix|p2p-capability|p2p-mailbox|p2p-owner-lut|support-rank)" >&2
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
