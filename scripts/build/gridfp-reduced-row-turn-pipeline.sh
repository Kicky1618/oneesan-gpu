#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ARCH="${ARCH:-native}"
SRC="$(repo_path src/cuda/gridfp/gridfp_reduced_production_row_turn_pipeline_microprobe.cu)"
OUT="$(build_path "${OUT:-gridfp_reduced_row_turn_pipeline}")"
PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"
PTXAS_FLAGS=()
[[ "$PTXAS_VERBOSE" == 1 ]] && PTXAS_FLAGS+=("-Xptxas=-v")

TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" \
  "${PTXAS_FLAGS[@]}" "$SRC" -o "$OUT"

echo "built $OUT"
