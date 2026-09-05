#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; NGPU="${NGPU:-8}"
TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
ARCH="${ARCH:-sm_103}"; PM_ACCUM="${PM_ACCUM:-1}"; WINDOW4="${WINDOW4:-1}"
COL_ILP="${COL_ILP:-2}"; PAIR_MLP="${PAIR_MLP:-1}"; QUAD_MLP="${QUAD_MLP:-0}"; DEPTHMAJOR="${DEPTHMAJOR:-1}"
DIRECTGATHER64="${DIRECTGATHER64:-1}"; SPARSE64="${SPARSE64:-1}"; SORTED="${SORTED:-1}"
CPASYNC_PAIR="${CPASYNC_PAIR:-1}"; CPASYNC_LOCAL_PAIR="${CPASYNC_LOCAL_PAIR:-0}"
OVERLAP_LOCAL="${OVERLAP_LOCAL:-0}"; OVERLAP_PIPE2="${OVERLAP_PIPE2:-0}"; OVERLAP_LOCAL_CG="${OVERLAP_LOCAL_CG:-0}"
THREADS="${THREADS:-256}"; HIGH_GX="${HIGH_GX:-32}"; HIGH_GY="${HIGH_GY:-8}"
LOW_GX="${LOW_GX:-16}"; LOW_GY="${LOW_GY:-8}"
DEVICE="${DEVICE:-0}"; LAUNCH_SKIP="${LAUNCH_SKIP:-0}"
for x in PM_ACCUM WINDOW4 PAIR_MLP QUAD_MLP DEPTHMAJOR DIRECTGATHER64 SPARSE64 SORTED CPASYNC_PAIR CPASYNC_LOCAL_PAIR OVERLAP_LOCAL OVERLAP_PIPE2 OVERLAP_LOCAL_CG; do
  v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0 or 1" >&2; exit 2; }
done
if [[ "$SPARSE64" == 1 && "$DIRECTGATHER64" != 1 ]]; then echo "SPARSE64 requires DIRECTGATHER64=1" >&2; exit 2; fi
if [[ "$CPASYNC_PAIR" == 1 && "$PAIR_MLP" != 1 ]]; then echo "CPASYNC_PAIR requires PAIR_MLP=1" >&2; exit 2; fi
if [[ "$CPASYNC_LOCAL_PAIR" == 1 && "$CPASYNC_PAIR" != 1 ]]; then echo "CPASYNC_LOCAL_PAIR requires CPASYNC_PAIR=1" >&2; exit 2; fi
if [[ "$OVERLAP_LOCAL" == 1 ]]; then
  [[ "$CPASYNC_PAIR" == 1 && "$PAIR_MLP" == 1 && "$DIRECTGATHER64" == 1 ]] || { echo "OVERLAP_LOCAL requires CPASYNC_PAIR=PAIR_MLP=DIRECTGATHER64=1" >&2; exit 2; }
  [[ "$CPASYNC_LOCAL_PAIR" == 0 ]] || { echo "OVERLAP_LOCAL is isolated from CPASYNC_LOCAL_PAIR" >&2; exit 2; }
fi
if [[ "$OVERLAP_PIPE2" == 1 && "$OVERLAP_LOCAL" != 1 ]]; then echo "OVERLAP_PIPE2 requires OVERLAP_LOCAL=1" >&2; exit 2; fi
if [[ "$OVERLAP_LOCAL_CG" == 1 && "$OVERLAP_LOCAL" != 1 ]]; then echo "OVERLAP_LOCAL_CG requires OVERLAP_LOCAL=1" >&2; exit 2; fi
if [[ "$QUAD_MLP" == 1 ]]; then
  [[ "$COL_ILP" == 4 && "$DIRECTGATHER64" == 1 && "$PAIR_MLP" == 1 ]] || { echo "QUAD_MLP requires COL_ILP=4 DIRECTGATHER64=PAIR_MLP=1" >&2; exit 2; }
  [[ "$CPASYNC_PAIR" == 0 && "$CPASYNC_LOCAL_PAIR" == 0 && "$OVERLAP_LOCAL" == 0 ]] || { echo "QUAD_MLP is isolated from cp.async pair modes" >&2; exit 2; }
fi
command -v ncu >/dev/null || { echo "ncu (Nsight Compute CLI) is required" >&2; exit 2; }
command -v nvcc >/dev/null || { echo "nvcc required" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi required" >&2; exit 2; }

TAG="c${COL_ILP}_p${PAIR_MLP}_q${QUAD_MLP}_cpa${CPASYNC_PAIR}_local${CPASYNC_LOCAL_PAIR}_ov${OVERLAP_LOCAL}_p2${OVERLAP_PIPE2}_cg${OVERLAP_LOCAL_CG}_sp${SPARSE64}_s${SORTED}"
BIN="${BIN:-$ONEESAN_BUILD_DIR/b300_directgather_ncu_n${N}_${TAG}}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_directgather_ncu_n${N}_${TAG}}"
mkdir -p "$(dirname "$PREFIX")"

N="$N" ARCH="$ARCH" OUT="$BIN" COL_ILP="$COL_ILP" DEPTHMAJOR="$DEPTHMAJOR" \
  FORCE7=0 MLP_WINDOW4="$WINDOW4" PAIR_MLP="$PAIR_MLP" QUAD_MLP="$QUAD_MLP" PREFETCH_NEXT=0 \
  DIRECTGATHER64="$DIRECTGATHER64" DIRECTGATHER_SPARSE64="$SPARSE64" SORTED="$SORTED" \
  CPASYNC_PAIR="$CPASYNC_PAIR" CPASYNC_LOCAL_PAIR="$CPASYNC_LOCAL_PAIR" \
  CPASYNC_OVERLAP_LOCAL_PAIR="$OVERLAP_LOCAL" CPASYNC_OVERLAP_LOCAL_PIPE2="$OVERLAP_PIPE2" \
  OVERLAP_LOCAL_CG="$OVERLAP_LOCAL_CG" PM_ACCUM="$PM_ACCUM" \
  PRECTX_FORWARD=0 PRECTX_REVERSE=0 MAXRREGCOUNT="${MAXRREGCOUNT:-0}" PTXAS_VERBOSE=1 \
  bash "$ONEESAN_ROOT/scripts/build/b300-directgather-colilp-fast.sh" \
  >"${PREFIX}.build.out" 2>"${PREFIX}.build.err"

# Profile only GPU0 and a single HIGH graph kernel replay. nvidia-smi's
# utilization.memory is only a trend signal; these sections expose actual
# DRAM/L2 throughput, achieved occupancy, issue activity and long-scoreboard
# stalls so we can distinguish insufficient MLP from register/SM bottlenecks.
BUCKET_THREADS="$THREADS" BUCKET_GRID_X="$LOW_GX" BUCKET_GRID_Y="$LOW_GY" \
BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY" \
BUCKET_HIGH_GRID_X="$HIGH_GX" BUCKET_HIGH_GRID_Y="$HIGH_GY" \
  ncu --devices "$DEVICE" --filter-mode per-gpu \
      --kernel-name-base function \
      --kernel-name 'regex:.*rankformula_nometa4_abstract_kernel.*' \
      --launch-skip "$LAUNCH_SKIP" --launch-count 1 \
      --section SpeedOfLight --section Occupancy --section WarpStateStats \
      --section MemoryWorkloadAnalysis \
      --page details --force-overwrite -o "$PREFIX" \
      "$BIN" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" \
      >"${PREFIX}.out" 2>"${PREFIX}.err"

echo "Nsight report: ${PREFIX}.ncu-rep" >&2
echo "config: tag=$TAG window4=$WINDOW4 depthmajor=$DEPTHMAJOR pm_accum=$PM_ACCUM high_grid=${HIGH_GX}x${HIGH_GY}" >&2
echo "Focus on: DRAM Throughput, L2 Throughput, Achieved Occupancy, Issue Active, Long Scoreboard, and register spills." >&2
