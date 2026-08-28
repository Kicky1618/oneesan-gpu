#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ARCH="${ARCH:-native}"
PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"
BASE_SRC="$(repo_path src/cuda/gridfp/gridfp_reduced_production_p2p_cycle_owner_descriptorless_microprobe.cu)"
PIPE_SRC="$(repo_path src/cuda/gridfp/gridfp_reduced_production_p2p_cycle_owner_pipeline_microprobe.cu)"
SRC_DIR="$(dirname "$BASE_SRC")"
GEN_DIR="$(dirname "$(build_path x)")"
GEN_BASE="$(build_path gridfp_reduced_production_p2p_cycle_owner_balanced_event_base_generated.cu)"
GEN_PIPE="$(build_path gridfp_reduced_production_p2p_cycle_owner_balanced_event_pipeline_base_generated.cu)"
GEN_EVENT="$(build_path gridfp_reduced_production_p2p_cycle_owner_balanced_event_generated.cu)"
BAL_GEN="$(repo_path scripts/build/generate-gridfp-cycle-owner-balanced.py)"
PIPE_GEN="$(repo_path scripts/build/generate-gridfp-cycle-owner-balanced-pipeline.py)"
EVENT_GEN="$(repo_path scripts/build/generate-gridfp-cycle-owner-balanced-event.py)"
OUT="$(build_path "${OUT:-gridfp_reduced_component_p2p-cycle-owner-balanced-event-pipeline}")"

python3 "$BAL_GEN" "$BASE_SRC" "$GEN_BASE"
python3 "$PIPE_GEN" "$PIPE_SRC" "$GEN_BASE" "$GEN_PIPE"
python3 "$EVENT_GEN" "$GEN_PIPE" "$GEN_EVENT"

grep -q 'gridfp-p2p-cycle-owner-balanced-event-pipeline' "$GEN_EVENT"
grep -q 'leader_batch_hash=1' "$GEN_EVENT"
grep -q 'batch_hash_shift=12' "$GEN_EVENT"
grep -q 'host_batch_barriers=0' "$GEN_EVENT"
grep -q 'event_schedule=staged' "$GEN_EVENT"
if sed -n '/Balanced canonical-leader batches/,/cycle-owner pipeline final sync set device/p' "$GEN_EVENT" | grep -q 'cudaStreamSynchronize'; then
  echo "balanced event runtime scheduler still contains host synchronization" >&2
  exit 3
fi

PTXAS_FLAGS=()
if [[ "$PTXAS_VERBOSE" == 1 ]]; then PTXAS_FLAGS+=("-Xptxas=-v"); fi
TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" \
  -I"$SRC_DIR" -I"$GEN_DIR" \
  "${PTXAS_FLAGS[@]}" "$GEN_EVENT" -o "$OUT"

echo "built $OUT (balanced_event_pipeline=1 schedule=staged batch_hash_shift=12 arch=$ARCH)"
