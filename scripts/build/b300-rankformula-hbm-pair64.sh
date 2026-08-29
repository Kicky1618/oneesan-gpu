#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

# Low-MC B300 preset: keep the two-column WINDOW4 schedule, but compress the
# common CROSS descriptor from uint4 (16 B) to one 64-bit word.  A second 64-bit
# rare descriptor is fetched only when a column has more than three sources.
N="${N:-27}"
OUT="${OUT:-$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_rankformula_hbm_pair64_n${N}}"

N="$N" OUT="$OUT" \
COL_ILP="${COL_ILP:-2}" \
DEPTHMAJOR=1 \
PAIR_MLP=1 \
MLP_WINDOW4=1 \
DIRECTGATHER64=1 \
CPASYNC_PAIR=0 \
PREFETCH_NEXT=0 \
FORCE7=0 \
SORTED=0 \
PM_ACCUM="${PM_ACCUM:-1}" \
MAXRREGCOUNT="${MAXRREGCOUNT:-0}" \
PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}" \
TRANSPOSE_MODE="${TRANSPOSE_MODE:-pipeline}" \
bash "$ONEESAN_ROOT/scripts/build/b300-directgather-colilp-fast.sh"

echo "b300-rankformula-hbm-pair64 OK out=$OUT n=$N col_ilp=${COL_ILP:-2} directgather64=1 pair_mlp=1 window4=1 cpasync_pair=0" >&2
