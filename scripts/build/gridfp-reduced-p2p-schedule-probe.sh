#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

MODE="${MODE:-compiled}"
ARCH="${ARCH:-native}"
PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"

case "$MODE" in
  baseline)
    SRC_REL="src/cuda/gridfp/gridfp_reduced_production_p2p_cycle_microprobe.cu"
    ;;
  ownerfirst)
    SRC_REL="src/cuda/gridfp/gridfp_reduced_production_p2p_ownerfirst_ab_microprobe.cu"
    ;;
  worklist)
    SRC_REL="src/cuda/gridfp/gridfp_reduced_production_p2p_worklist_microprobe.cu"
    ;;
  compiled)
    SRC_REL="src/cuda/gridfp/gridfp_reduced_production_p2p_compiled_schedule_microprobe.cu"
    ;;
  *)
    echo "invalid MODE=$MODE (baseline|ownerfirst|worklist|compiled)" >&2
    exit 2
    ;;
esac

SRC="$(repo_path "$SRC_REL")"
OUT="$(build_path "${OUT:-gridfp_reduced_p2p_${MODE}}")"
PTXAS_FLAGS=()
if [[ "$PTXAS_VERBOSE" == 1 ]]; then PTXAS_FLAGS+=("-Xptxas=-v"); fi

TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" \
  "${PTXAS_FLAGS[@]}" "$SRC" -o "$OUT"

echo "built $OUT (mode=$MODE arch=$ARCH)"
