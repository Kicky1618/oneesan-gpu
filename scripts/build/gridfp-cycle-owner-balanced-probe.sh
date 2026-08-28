#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ARCH="${ARCH:-native}"
PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"
BASE_SRC="$(repo_path src/cuda/gridfp/gridfp_reduced_production_p2p_cycle_owner_descriptorless_microprobe.cu)"
SRC_DIR="$(dirname "$BASE_SRC")"
GENERATOR="$(repo_path scripts/build/generate-gridfp-cycle-owner-balanced.py)"
GEN_SRC="$(build_path gridfp_reduced_production_p2p_cycle_owner_balanced_descriptorless_generated.cu)"
OUT="$(build_path "${OUT:-gridfp_reduced_component_p2p-cycle-owner-balanced-descriptorless}")"

python3 "$GENERATOR" "$BASE_SRC" "$GEN_SRC"

# Generation-time sanity checks keep the benchmark from silently compiling the
# old invariant-hash source if upstream formatting changes.
grep -q 'cycle_owner_balanced_host_batch_id' "$GEN_SRC"
grep -q 'cycle_owner_balanced_device_batch_id' "$GEN_SRC"
grep -q 'gridfp-p2p-cycle-owner-balanced-descriptorless' "$GEN_SRC"
grep -q 'batch_hash_shift=12' "$GEN_SRC"
if grep -q 'hostlist_batch_id(' "$GEN_SRC"; then
  echo "balanced generator left hostlist_batch_id call" >&2
  exit 3
fi
if grep -q 'persistent_batch_id(' "$GEN_SRC"; then
  echo "balanced generator left persistent_batch_id call" >&2
  exit 4
fi

PTXAS_FLAGS=()
if [[ "$PTXAS_VERBOSE" == 1 ]]; then PTXAS_FLAGS+=("-Xptxas=-v"); fi
TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" \
  -I"$SRC_DIR" "${PTXAS_FLAGS[@]}" "$GEN_SRC" -o "$OUT"

echo "built $OUT (cycle_owner_balanced=1 batch_hash_shift=12 arch=$ARCH generated=$GEN_SRC)"
