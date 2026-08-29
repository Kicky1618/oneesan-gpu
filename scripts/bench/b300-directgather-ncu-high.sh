#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; NGPU="${NGPU:-8}"
TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
ARCH="${ARCH:-native}"; PM_ACCUM="${PM_ACCUM:-0}"; WINDOW4="${WINDOW4:-0}"
THREADS="${THREADS:-256}"; HIGH_GX="${HIGH_GX:-128}"; HIGH_GY="${HIGH_GY:-1}"
LOW_GX="${LOW_GX:-16}"; LOW_GY="${LOW_GY:-8}"
DEVICE="${DEVICE:-0}"; LAUNCH_SKIP="${LAUNCH_SKIP:-0}"
command -v ncu >/dev/null || { echo "ncu (Nsight Compute CLI) is required" >&2; exit 2; }
command -v nvcc >/dev/null || { echo "nvcc required" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi required" >&2; exit 2; }

BIN="${BIN:-$ONEESAN_BUILD_DIR/b300_directgather_ncu_n${N}_w${WINDOW4}_pm${PM_ACCUM}}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_directgather_ncu_n${N}_w${WINDOW4}_pm${PM_ACCUM}}"
mkdir -p "$(dirname "$PREFIX")"

N="$N" ARCH="$ARCH" OUT="$BIN" \
  RANKFORMULA_NOMETA_BLOCK=16 RANKFORMULA_NOMETA_WARPSHARE=1 \
  RANKFORMULA_NOMETA_COOPGROUP=1 RANKFORMULA_NOMETA_COOP_UNROLL=0 \
  RANKFORMULA_NOMETA_GROUP56=0 RANKFORMULA_NOMETA_GROUP61=1 \
  RANKFORMULA_NOMETA_DIRECTMAP=1 RANKFORMULA_DIRECTGATHER=1 \
  RANKFORMULA_DIRECTGATHER_FORCE7=0 RANKFORMULA_MLP_WINDOW4="$WINDOW4" \
  RANKFORMULA_ABSTRACT_SELECT8=1 RANKFORMULA_ABSTRACT_DEPTH4=1 \
  RANKFORMULA_ABSTRACT_SRCPACK10=1 RANKFORMULA_GATHER_MLP=1 \
  DEPTHCODE_DECODE_LOAD=ldg RANKSTREAM_LUT_LOAD=ldg \
  PM_ACCUM="$PM_ACCUM" PTXAS_VERBOSE=1 TRANSPOSE_MODE=pipeline \
  bash "$ONEESAN_ROOT/scripts/build/b300-bucket-snake-pattern10-depthcode-rankformula-nometa4-abstract.sh" \
  >"${PREFIX}.build.out" 2>"${PREFIX}.build.err"

# Profile only GPU0 and only one matching HIGH kernel.  This avoids collecting
# every graph launch on all eight B300s while still exposing the limiting SM/L2/
# DRAM/warp-stall counters.  Nsight Compute may replay the selected launch.
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
echo "Focus on: DRAM throughput, L2 throughput, achieved occupancy, issue-active, and Long Scoreboard stalls." >&2
