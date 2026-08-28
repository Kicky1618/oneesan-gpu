#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ARCH="${ARCH:-native}"
PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"
VARIANT="${VARIANT:-segment}"
case "$VARIANT" in
  segment)
    SRC_REL="src/cuda/gridfp/gridfp_reduced_production_p2p_host_persistent_pipeline_microprobe.cu"
    DEFAULT_OUT="gridfp_reduced_component_p2p-host-persistent-pipeline"
    ;;
  cycle-owner)
    SRC_REL="src/cuda/gridfp/gridfp_reduced_production_p2p_cycle_owner_pipeline_microprobe.cu"
    DEFAULT_OUT="gridfp_reduced_component_p2p-cycle-owner-pipeline"
    ;;
  *)
    echo "invalid VARIANT=$VARIANT (segment|cycle-owner)" >&2
    exit 2
    ;;
esac
SRC="$(repo_path "$SRC_REL")"
OUT="$(build_path "${OUT:-$DEFAULT_OUT}")"
PTXAS_FLAGS=()
if [[ "$PTXAS_VERBOSE" == 1 ]]; then PTXAS_FLAGS+=("-Xptxas=-v"); fi

TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" \
  "${PTXAS_FLAGS[@]}" "$SRC" -o "$OUT"

echo "built $OUT (persistent_pipeline=1 variant=$VARIANT arch=$ARCH)"
