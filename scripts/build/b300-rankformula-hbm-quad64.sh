#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

# Four-column B300 experiment on compact 64-bit direct-gather descriptors.
# Register/dense remains the default. Sparse64, quad cp.async, and forward/
# reverse PRECTX are explicit opt-ins so measured winners can be reproduced
# without changing the production preset.
N="${N:-27}"
PRECTX_FORWARD="${PRECTX_FORWARD:-0}"
PRECTX_REVERSE="${PRECTX_REVERSE:-0}"
DIRECTGATHER_SPARSE64="${DIRECTGATHER_SPARSE64:-0}"
CPASYNC_PAIR="${CPASYNC_PAIR:-0}"
for x in PRECTX_FORWARD PRECTX_REVERSE DIRECTGATHER_SPARSE64 CPASYNC_PAIR; do
  v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0 or 1" >&2; exit 2; }
done
OUT="${OUT:-$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_rankformula_hbm_quad64_sp${DIRECTGATHER_SPARSE64}_cpa${CPASYNC_PAIR}_f${PRECTX_FORWARD}r${PRECTX_REVERSE}_n${N}}"

N="$N" OUT="$OUT" \
COL_ILP=4 \
DEPTHMAJOR=1 \
PAIR_MLP=1 \
QUAD_MLP=1 \
MLP_WINDOW4=1 \
DIRECTGATHER64=1 \
DIRECTGATHER_SPARSE64="$DIRECTGATHER_SPARSE64" \
CPASYNC_PAIR="$CPASYNC_PAIR" \
CPASYNC_LOCAL_PAIR=0 \
CPASYNC_OVERLAP_LOCAL_PAIR=0 \
PRECTX_FORWARD="$PRECTX_FORWARD" \
PRECTX_REVERSE="$PRECTX_REVERSE" \
PREFETCH_NEXT=0 \
FORCE7=0 \
SORTED=0 \
PM_ACCUM="${PM_ACCUM:-1}" \
MAXRREGCOUNT="${MAXRREGCOUNT:-0}" \
PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}" \
TRANSPOSE_MODE="${TRANSPOSE_MODE:-pipeline}" \
bash "$ONEESAN_ROOT/scripts/build/b300-directgather-colilp-fast.sh"

echo "b300-rankformula-hbm-quad64 OK out=$OUT n=$N col_ilp=4 directgather64=1 sparse64=$DIRECTGATHER_SPARSE64 pair_mlp=1 quad_mlp=1 prectx_forward=$PRECTX_FORWARD prectx_reverse=$PRECTX_REVERSE cpasync_pair=$CPASYNC_PAIR" >&2
