#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-27}"
COL_ILP="${COL_ILP:-2}"
MLP_WINDOW4="${MLP_WINDOW4:-1}"
OUT="${OUT:-oneesan_cuda_gridfp_b300_directgather64_colilp${COL_ILP}_n${N}}"

N="$N" OUT="$OUT" ARCH="${ARCH:-native}" \
  LOW_LUT_K="${LOW_LUT_K:-$(((N + 1) / 2))}" \
  TRANSPOSE_MODE="${TRANSPOSE_MODE:-pipeline}" \
  COL_ILP="$COL_ILP" DEPTHMAJOR=1 DIRECTGATHER64=1 \
  FORCE7=0 MLP_WINDOW4="$MLP_WINDOW4" PAIR_MLP=0 PREFETCH_NEXT=0 \
  PM_ACCUM="${PM_ACCUM:-1}" MAXRREGCOUNT="${MAXRREGCOUNT:-0}" \
  PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}" \
  bash "$ONEESAN_ROOT/scripts/build/b300-directgather-colilp-fast.sh"

echo "preset=directgather64-fast n=$N col_ilp=$COL_ILP depthmajor=1 directgather64=1 window4=$MLP_WINDOW4 pm_accum=${PM_ACCUM:-1}" >&2
echo "run example:" >&2
echo "  BUCKET_THREADS=${BUCKET_THREADS:-256} BUCKET_GRID_X=${BUCKET_GRID_X:-32} BUCKET_GRID_Y=${BUCKET_GRID_Y:-8} \\" >&2
echo "    $ONEESAN_BUILD_DIR/$OUT $N <target_mib> <max_window> 8 <mod...>" >&2
