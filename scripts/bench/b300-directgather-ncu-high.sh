#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; NGPU="${NGPU:-8}"
TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
ARCH="${ARCH:-native}"; PM_ACCUM="${PM_ACCUM:-1}"; WINDOW4="${WINDOW4:-1}"
COL_ILP="${COL_ILP:-2}"; PAIR_MLP="${PAIR_MLP:-0}"; DEPTHMAJOR="${DEPTHMAJOR:-1}"
DIRECTGATHER64="${DIRECTGATHER64:-1}"
THREADS="${THREADS:-256}"; HIGH_GX="${HIGH_GX:-32}"; HIGH_GY="${HIGH_GY:-8}"
LOW_GX="${LOW_GX:-16}"; LOW_GY="${LOW_GY:-8}"
DEVICE="${DEVICE:-0}"; LAUNCH_SKIP="${LAUNCH_SKIP:-0}"
for x in PM_ACCUM WINDOW4 PAIR_MLP DEPTHMAJOR DIRECTGATHER64; do
  v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0 or 1" >&2; exit 2; }
done
if [[ "$DIRECTGATHER64" == 1 && "$PAIR_MLP" == 1 ]]; then
  echo "DIRECTGATHER64 and PAIR_MLP are isolated A/B modes; choose one" >&2; exit 2
fi
command -v ncu >/dev/null || { echo "ncu (Nsight Compute CLI) is required" >&2; exit 2; }
command -v nvcc >/dev/null || { echo "nvcc required" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi required" >&2; exit 2; }

BIN="${BIN:-$ONEESAN_BUILD_DIR/b300_directgather_ncu_n${N}_c${COL_ILP}_p${PAIR_MLP}_d64${DIRECTGATHER64}}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_directgather_ncu_n${N}_c${COL_ILP}_p${PAIR_MLP}_d64${DIRECTGATHER64}}"
mkdir -p "$(dirname "$PREFIX")"

N="$N" ARCH="$ARCH" OUT="$BIN" COL_ILP="$COL_ILP" DEPTHMAJOR="$DEPTHMAJOR" \
  FORCE7=0 MLP_WINDOW4="$WINDOW4" PAIR_MLP="$PAIR_MLP" PREFETCH_NEXT=0 \
  DIRECTGATHER64="$DIRECTGATHER64" PM_ACCUM="$PM_ACCUM" \
  MAXRREGCOUNT="${MAXRREGCOUNT:-0}" PTXAS_VERBOSE=1 \
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
echo "config: col_ilp=$COL_ILP pair_mlp=$PAIR_MLP directgather64=$DIRECTGATHER64 window4=$WINDOW4 depthmajor=$DEPTHMAJOR pm_accum=$PM_ACCUM high_grid=${HIGH_GX}x${HIGH_GY}" >&2
echo "Focus on: DRAM throughput, L2 throughput, achieved occupancy, issue-active, and Long Scoreboard stalls." >&2
