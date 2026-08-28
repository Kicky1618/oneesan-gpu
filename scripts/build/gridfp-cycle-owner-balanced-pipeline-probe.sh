#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ARCH="${ARCH:-native}"
PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"
BASE_SRC="$(repo_path src/cuda/gridfp/gridfp_reduced_production_p2p_cycle_owner_descriptorless_microprobe.cu)"
PIPE_SRC="$(repo_path src/cuda/gridfp/gridfp_reduced_production_p2p_cycle_owner_pipeline_microprobe.cu)"
SRC_DIR="$(dirname "$BASE_SRC")"
GEN_BASE="$(build_path gridfp_reduced_production_p2p_cycle_owner_balanced_base_generated.cu)"
GEN_PIPE="$(build_path gridfp_reduced_production_p2p_cycle_owner_balanced_pipeline_generated.cu)"
BAL_GEN="$(repo_path scripts/build/generate-gridfp-cycle-owner-balanced.py)"
PIPE_GEN="$(repo_path scripts/build/generate-gridfp-cycle-owner-balanced-pipeline.py)"
OUT="$(build_path "${OUT:-gridfp_reduced_component_p2p-cycle-owner-balanced-pipeline}")"

python3 "$BAL_GEN" "$BASE_SRC" "$GEN_BASE"
python3 "$PIPE_GEN" "$PIPE_SRC" "$GEN_BASE" "$GEN_PIPE"

grep -q 'gridfp-p2p-cycle-owner-balanced-pipeline' "$GEN_PIPE"
grep -q 'leader_batch_hash=1' "$GEN_PIPE"
grep -q 'batch_hash_shift=12' "$GEN_PIPE"
grep -q "#include \"$(basename "$GEN_BASE")\"" "$GEN_PIPE"

PTXAS_FLAGS=()
if [[ "$PTXAS_VERBOSE" == 1 ]]; then PTXAS_FLAGS+=("-Xptxas=-v"); fi
TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" \
  -I"$SRC_DIR" -I"$(dirname "$GEN_BASE")" \
  "${PTXAS_FLAGS[@]}" "$GEN_PIPE" -o "$OUT"

echo "built $OUT (cycle_owner_balanced_pipeline=1 batch_hash_shift=12 arch=$ARCH)"
