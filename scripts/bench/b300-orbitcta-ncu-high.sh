#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; NGPU="${NGPU:-8}"
TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
ARCH="${ARCH:-native}"; PM_ACCUM="${PM_ACCUM:-0}"; WINDOW4="${WINDOW4:-0}"
THREADS="${THREADS:-256}"; ORBIT_GY="${ORBIT_GY:-128}"
LOW_GX="${LOW_GX:-16}"; LOW_GY="${LOW_GY:-8}"
DEVICE="${DEVICE:-0}"; LAUNCH_SKIP="${LAUNCH_SKIP:-0}"
command -v ncu >/dev/null || { echo "ncu required" >&2; exit 2; }
command -v nvcc >/dev/null || { echo "nvcc required" >&2; exit 2; }

BIN="${BIN:-$ONEESAN_BUILD_DIR/b300_orbitcta_ncu_n${N}}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_orbitcta_ncu_n${N}_t${THREADS}_y${ORBIT_GY}}"
mkdir -p "$(dirname "$PREFIX")"

N="$N" ARCH="$ARCH" OUT="$BIN" PM_ACCUM="$PM_ACCUM" \
  RANKFORMULA_MLP_WINDOW4="$WINDOW4" PTXAS_VERBOSE=1 \
  bash "$ONEESAN_ROOT/scripts/build/b300-directgather-orbitcta.sh" \
  >"${PREFIX}.build.out" 2>"${PREFIX}.build.err"

BUCKET_THREADS="$THREADS" BUCKET_GRID_X="$LOW_GX" BUCKET_GRID_Y="$LOW_GY" \
BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY" \
BUCKET_ORBITCTA_GRID_Y="$ORBIT_GY" \
  ncu --devices "$DEVICE" --filter-mode per-gpu \
      --kernel-name-base function \
      --kernel-name 'regex:.*orbitcta_kernel.*' \
      --launch-skip "$LAUNCH_SKIP" --launch-count 1 \
      --section SpeedOfLight --section Occupancy --section WarpStateStats \
      --section MemoryWorkloadAnalysis \
      --page details --force-overwrite -o "$PREFIX" \
      "$BIN" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" \
      >"${PREFIX}.out" 2>"${PREFIX}.err"

echo "Nsight report: ${PREFIX}.ncu-rep" >&2
echo "config: scheduler=orbitcta threads=$THREADS orbit_gy=$ORBIT_GY depthmajor=1 window4=$WINDOW4 pm_accum=$PM_ACCUM" >&2
echo "compare DRAM/L2 throughput, issue active, achieved occupancy, and Long Scoreboard against b300-directgather-ncu-high.sh" >&2
