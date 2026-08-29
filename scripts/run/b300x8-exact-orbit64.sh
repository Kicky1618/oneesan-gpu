#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"
if (( $# > 0 )); then shift; fi
NGPU="${NGPU:-8}"
TARGET_MIB="${TARGET_MIB:-16384}"
MAX_WINDOW="${MAX_WINDOW:-14}"
ARCH="${ARCH:-native}"
THREADS="${BUCKET_THREADS:-256}"
ORBIT_GY="${BUCKET_ORBITCTA_GRID_Y:-128}"
LOW_GX="${BUCKET_LOW_GRID_X:-16}"
LOW_GY="${BUCKET_LOW_GRID_Y:-8}"
TRANSPOSE_CHUNK_MIB="${BUCKET_TRANSPOSE_CHUNK_MIB:-1024}"
BUCKET_RESERVE_MIB="${BUCKET_RESERVE_MIB:-8192}"
DIRECTGATHER_SPARSE64="${DIRECTGATHER_SPARSE64:-0}"
PREFLIGHT="${PREFLIGHT:-1}"
REBUILD="${REBUILD:-0}"
SUFFIX="dense64"; [[ "$DIRECTGATHER_SPARSE64" == 1 ]] && SUFFIX="sparse64"
BIN="${BIN:-$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_orbit64_${SUFFIX}_batch_n${N}}"
WORK_DIR="${WORK_DIR:-$ONEESAN_ROOT/work/b300_exact_orbit64_${SUFFIX}_n${N}}"

[[ "$NGPU" == 8 ]] || { echo "orbit64 production backend currently requires NGPU=8" >&2; exit 2; }
for x in REBUILD PREFLIGHT DIRECTGATHER_SPARSE64; do v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0 or 1" >&2; exit 2; }; done
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found" >&2; exit 2; }
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
(( visible >= 8 )) || { echo "need 8 visible GPUs, got $visible" >&2; exit 2; }

if [[ "$REBUILD" == 1 || ! -x "$BIN" ]]; then
  echo "building orbit64 exact backend n=$N sparse64=$DIRECTGATHER_SPARSE64 threads=$THREADS orbit_gy=$ORBIT_GY low=${LOW_GX}x${LOW_GY}" >&2
  N="$N" ARCH="$ARCH" OUT="$BIN" DIRECTGATHER64=1 DIRECTGATHER_SPARSE64="$DIRECTGATHER_SPARSE64" PTXAS_VERBOSE=1 \
    bash "$ONEESAN_ROOT/scripts/build/b300-directgather-orbitcta.sh"
fi

export BUCKET_THREADS="$THREADS"
export BUCKET_ORBITCTA_GRID_Y="$ORBIT_GY"
export BUCKET_GRID_X="$LOW_GX"
export BUCKET_GRID_Y="$LOW_GY"
export BUCKET_LOW_GRID_X="$LOW_GX"
export BUCKET_LOW_GRID_Y="$LOW_GY"
export BUCKET_TRANSPOSE_CHUNK_MIB="$TRANSPOSE_CHUNK_MIB"
export BUCKET_RESERVE_MIB

if [[ "$PREFLIGHT" == 1 ]]; then
  PLAN_LOG="${WORK_DIR}.plan.log"
  mkdir -p "$(dirname "$PLAN_LOG")"
  echo "=== orbit64 $SUFFIX plan-only preflight ===" >&2
  "$BIN" "$N" "$TARGET_MIB" "$MAX_WINDOW" 8 --plan-only 2> >(tee "$PLAN_LOG" >&2)
  plan="$(grep 'backend=gridfp-b300-bucket-snake-onepass-graph-batch-v0.1-plan' "$PLAN_LOG" | tail -n1 || true)"
  [[ -n "$plan" ]] || { echo "orbit64 plan line missing" >&2; exit 3; }
  echo "=== current HBM free ===" >&2
  nvidia-smi --query-gpu=index,memory.total,memory.free --format=csv,noheader >&2 || true
fi

echo "exact orbit64 n=$N gpus=8 sparse64=$DIRECTGATHER_SPARSE64 binary=$BIN work_dir=$WORK_DIR" >&2
echo "threads=$BUCKET_THREADS orbit_gy=$BUCKET_ORBITCTA_GRID_Y low_grid=${BUCKET_LOW_GRID_X}x${BUCKET_LOW_GRID_Y} transpose_chunk_mib=$BUCKET_TRANSPOSE_CHUNK_MIB reserve_mib=$BUCKET_RESERVE_MIB" >&2
exec python3 "$ONEESAN_ROOT/scripts/solve/solve_b300_exact_batch.py" "$N" \
  --binary "$BIN" \
  --target-mib "$TARGET_MIB" \
  --max-window "$MAX_WINDOW" \
  --gpus 8 \
  --work-dir "$WORK_DIR" \
  "$@"
