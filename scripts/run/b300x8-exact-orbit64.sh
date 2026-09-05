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
ORBITCTA_COL_ILP="${ORBITCTA_COL_ILP:-2}"
PAIR_MLP="${PAIR_MLP:-1}"
CPASYNC_PAIR="${CPASYNC_PAIR:-0}"
PRECTX_FORWARD="${PRECTX_FORWARD:-0}"
PRECTX_REVERSE="${PRECTX_REVERSE:-0}"
PM_ACCUM="${PM_ACCUM:-1}"
RANKFORMULA_MLP_WINDOW4="${RANKFORMULA_MLP_WINDOW4:-1}"
PREFLIGHT="${PREFLIGHT:-1}"
REBUILD="${REBUILD:-0}"
SPARSE_TAG="dense64"; [[ "$DIRECTGATHER_SPARSE64" == 1 ]] && SPARSE_TAG="sparse64"
CFG_TAG="${SPARSE_TAG}_i${ORBITCTA_COL_ILP}_pair${PAIR_MLP}_cpa${CPASYNC_PAIR}_pf${PRECTX_FORWARD}_pr${PRECTX_REVERSE}_pm${PM_ACCUM}"
BIN="${BIN:-$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_orbit64_${CFG_TAG}_batch_n${N}}"
WORK_DIR="${WORK_DIR:-$ONEESAN_ROOT/work/b300_exact_orbit64_${CFG_TAG}_n${N}}"

[[ "$NGPU" == 8 ]] || { echo "orbit64 production backend currently requires NGPU=8" >&2; exit 2; }
for x in REBUILD PREFLIGHT DIRECTGATHER_SPARSE64 PAIR_MLP CPASYNC_PAIR PRECTX_FORWARD PRECTX_REVERSE PM_ACCUM RANKFORMULA_MLP_WINDOW4; do
  v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0 or 1" >&2; exit 2; }
done
case "$ORBITCTA_COL_ILP" in 1|2|4) ;; *) echo "ORBITCTA_COL_ILP must be 1,2,4" >&2; exit 2;; esac
if [[ "$PAIR_MLP" == 1 ]]; then
  [[ "$RANKFORMULA_MLP_WINDOW4" == 1 ]] || { echo "PAIR_MLP requires RANKFORMULA_MLP_WINDOW4=1" >&2; exit 2; }
  [[ "$ORBITCTA_COL_ILP" == 2 || "$ORBITCTA_COL_ILP" == 4 ]] || { echo "PAIR_MLP requires ORBITCTA_COL_ILP=2 or 4" >&2; exit 2; }
fi
[[ "$CPASYNC_PAIR" == 0 || "$PAIR_MLP" == 1 ]] || { echo "CPASYNC_PAIR requires PAIR_MLP=1" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found" >&2; exit 2; }
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
(( visible >= 8 )) || { echo "need 8 visible GPUs, got $visible" >&2; exit 2; }

if [[ "$CPASYNC_PAIR" == 1 ]]; then
  echo "=== cp.async remote-peer preflight ===" >&2
  CPA_LOG="${WORK_DIR}.cpasync-peer.log"
  mkdir -p "$(dirname "$CPA_LOG")"
  ARCH="$ARCH" NGPU=8 THREADS="$THREADS" \
    bash "$ONEESAN_ROOT/scripts/bench/b300-cpasync-remote-peer-microprobe.sh" \
    >"$CPA_LOG" 2>&1
  grep -q 'cp_async_remote_peer=OK exact=OK' "$CPA_LOG" || { cat "$CPA_LOG" >&2; echo "cp.async peer preflight failed" >&2; exit 4; }
fi

if [[ "$REBUILD" == 1 || ! -x "$BIN" ]]; then
  echo "building orbit64 exact n=$N cfg=$CFG_TAG threads=$THREADS orbit_gy=$ORBIT_GY low=${LOW_GX}x${LOW_GY}" >&2
  N="$N" ARCH="$ARCH" OUT="$BIN" DIRECTGATHER64=1 DIRECTGATHER_SPARSE64="$DIRECTGATHER_SPARSE64" \
    ORBITCTA_COL_ILP="$ORBITCTA_COL_ILP" PAIR_MLP="$PAIR_MLP" CPASYNC_PAIR="$CPASYNC_PAIR" \
    PRECTX_FORWARD="$PRECTX_FORWARD" PRECTX_REVERSE="$PRECTX_REVERSE" \
    RANKFORMULA_MLP_WINDOW4="$RANKFORMULA_MLP_WINDOW4" PM_ACCUM="$PM_ACCUM" PTXAS_VERBOSE=1 \
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
  echo "=== orbit64 $CFG_TAG plan-only preflight ===" >&2
  "$BIN" "$N" "$TARGET_MIB" "$MAX_WINDOW" 8 --plan-only 2> >(tee "$PLAN_LOG" >&2)
  plan="$(grep 'backend=gridfp-b300-bucket-snake-onepass-graph-batch-v0.1-plan' "$PLAN_LOG" | tail -n1 || true)"
  [[ -n "$plan" ]] || { echo "orbit64 plan line missing" >&2; exit 3; }
  echo "=== current HBM free ===" >&2
  nvidia-smi --query-gpu=index,memory.total,memory.free --format=csv,noheader >&2 || true
fi

echo "exact orbit64 n=$N gpus=8 cfg=$CFG_TAG binary=$BIN work_dir=$WORK_DIR" >&2
echo "threads=$BUCKET_THREADS orbit_gy=$BUCKET_ORBITCTA_GRID_Y low_grid=${BUCKET_LOW_GRID_X}x${BUCKET_LOW_GRID_Y} transpose_chunk_mib=$BUCKET_TRANSPOSE_CHUNK_MIB reserve_mib=$BUCKET_RESERVE_MIB" >&2
exec python3 "$ONEESAN_ROOT/scripts/solve/solve_b300_exact_batch.py" "$N" \
  --binary "$BIN" \
  --target-mib "$TARGET_MIB" \
  --max-window "$MAX_WINDOW" \
  --gpus 8 \
  --work-dir "$WORK_DIR" \
  "$@"
