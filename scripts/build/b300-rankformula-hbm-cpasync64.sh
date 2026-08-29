#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

# Experimental low-MC B300 path: two compressed descriptor columns are decoded
# first, then up to fourteen selected 32-bit CROSS values are staged through
# shared memory with cp.async. Sparse64 removes zero descriptors; SORTED orders
# each source-rank list ascending before the runtime gather.
N="${N:-27}"
SPARSE64="${SPARSE64:-1}"
SORTED="${SORTED:-0}"
for x in SPARSE64 SORTED; do
  v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0 or 1" >&2; exit 2; }
done
OUT="${OUT:-$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_rankformula_hbm_cpasync64_sp${SPARSE64}_sort${SORTED}_n${N}}"

N="$N" OUT="$OUT" \
COL_ILP="${COL_ILP:-2}" \
DEPTHMAJOR=1 \
PAIR_MLP=1 \
MLP_WINDOW4=1 \
DIRECTGATHER64=1 \
DIRECTGATHER_SPARSE64="$SPARSE64" \
CPASYNC_PAIR=1 \
PREFETCH_NEXT=0 \
FORCE7=0 \
SORTED="$SORTED" \
PM_ACCUM="${PM_ACCUM:-1}" \
MAXRREGCOUNT="${MAXRREGCOUNT:-0}" \
PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}" \
TRANSPOSE_MODE="${TRANSPOSE_MODE:-pipeline}" \
bash "$ONEESAN_ROOT/scripts/build/b300-directgather-colilp-fast.sh"

echo "b300-rankformula-hbm-cpasync64 OK out=$OUT n=$N col_ilp=${COL_ILP:-2} directgather64=1 sparse64=$SPARSE64 sorted=$SORTED pair_mlp=1 cpasync_pair=1" >&2
