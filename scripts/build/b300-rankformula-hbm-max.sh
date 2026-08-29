#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

# Aggressive B300 latency-hiding preset for the observed low memory-controller
# utilization regime. Depth-major makes warp descriptor reads contiguous,
# COL_ILP=2 keeps two columns in flight, and next-stripe L2 prefetch issues the
# following batch without extending the live value window.
N="${N:-27}"
COL_ILP="${COL_ILP:-2}"
FORCE7="${FORCE7:-0}"
MLP_WINDOW4="${MLP_WINDOW4:-0}"
PREFETCH_NEXT="${PREFETCH_NEXT:-1}"
OUT="${OUT:-$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_rankformula_hbm_max_n${N}}"

N="$N" OUT="$OUT" \
COL_ILP="$COL_ILP" \
DEPTHMAJOR="${DEPTHMAJOR:-1}" \
FORCE7="$FORCE7" \
MLP_WINDOW4="$MLP_WINDOW4" \
PREFETCH_NEXT="$PREFETCH_NEXT" \
PM_ACCUM="${PM_ACCUM:-1}" \
MAXRREGCOUNT="${MAXRREGCOUNT:-0}" \
PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}" \
TRANSPOSE_MODE="${TRANSPOSE_MODE:-pipeline}" \
bash "$ONEESAN_ROOT/scripts/build/b300-directgather-colilp-fast.sh"

echo "b300-rankformula-hbm-max OK out=$OUT n=$N col_ilp=$COL_ILP depthmajor=${DEPTHMAJOR:-1} prefetch_next=$PREFETCH_NEXT force7=$FORCE7 mlp_window4=$MLP_WINDOW4 maxrregcount=${MAXRREGCOUNT:-0}" >&2
