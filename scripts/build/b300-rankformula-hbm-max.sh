#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

# Aggressive B300 latency-hiding preset for the observed low memory-controller
# utilization regime. Depth-major keeps descriptor reads contiguous. Pair MLP
# issues useful source reads for two columns before either reduction. The common
# CROSS descriptor is compressed to 64 bits by default; sparse64 and HIGH prectx
# remain A/B switches until exact B300 measurements establish their wins.
N="${N:-27}"
COL_ILP="${COL_ILP:-2}"
PAIR_MLP="${PAIR_MLP:-1}"
CPASYNC_PAIR="${CPASYNC_PAIR:-0}"
CPASYNC_LOCAL_PAIR="${CPASYNC_LOCAL_PAIR:-0}"
CPASYNC_OVERLAP_LOCAL_PAIR="${CPASYNC_OVERLAP_LOCAL_PAIR:-0}"
CPASYNC_OVERLAP_LOCAL_PIPE2="${CPASYNC_OVERLAP_LOCAL_PIPE2:-0}"
OVERLAP_LOCAL_CG="${OVERLAP_LOCAL_CG:-0}"
SORTED="${SORTED:-0}"
PRECTX_FORWARD="${PRECTX_FORWARD:-0}"
PRECTX_REVERSE="${PRECTX_REVERSE:-0}"
FORCE7="${FORCE7:-0}"
MLP_WINDOW4="${MLP_WINDOW4:-1}"
PREFETCH_NEXT="${PREFETCH_NEXT:-0}"
DIRECTGATHER64="${DIRECTGATHER64:-1}"
DIRECTGATHER_SPARSE64="${DIRECTGATHER_SPARSE64:-0}"
OUT="${OUT:-$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_rankformula_hbm_max_n${N}}"

for x in PAIR_MLP CPASYNC_PAIR CPASYNC_LOCAL_PAIR CPASYNC_OVERLAP_LOCAL_PAIR CPASYNC_OVERLAP_LOCAL_PIPE2 OVERLAP_LOCAL_CG SORTED PRECTX_FORWARD PRECTX_REVERSE FORCE7 MLP_WINDOW4 PREFETCH_NEXT DIRECTGATHER64 DIRECTGATHER_SPARSE64; do
  v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0 or 1" >&2; exit 2; }
done
if [[ "$PAIR_MLP" == 1 ]]; then
  [[ "$COL_ILP" == 2 || "$COL_ILP" == 4 ]] || { echo "PAIR_MLP requires COL_ILP=2 or 4" >&2; exit 2; }
  [[ "$MLP_WINDOW4" == 1 ]] || { echo "PAIR_MLP requires MLP_WINDOW4=1" >&2; exit 2; }
  [[ "$FORCE7" == 0 ]] || { echo "PAIR_MLP and FORCE7 are isolated modes" >&2; exit 2; }
fi
if [[ "$CPASYNC_PAIR" == 1 ]]; then
  [[ "$PAIR_MLP" == 1 ]] || { echo "CPASYNC_PAIR requires PAIR_MLP=1" >&2; exit 2; }
  [[ "$PREFETCH_NEXT" == 0 ]] || { echo "CPASYNC_PAIR and PREFETCH_NEXT are isolated modes" >&2; exit 2; }
fi
if [[ "$CPASYNC_LOCAL_PAIR" == 1 && "$CPASYNC_PAIR" != 1 ]]; then
  echo "CPASYNC_LOCAL_PAIR requires CPASYNC_PAIR=1" >&2; exit 2
fi
if [[ "$CPASYNC_OVERLAP_LOCAL_PAIR" == 1 ]]; then
  [[ "$CPASYNC_PAIR" == 1 ]] || { echo "CPASYNC_OVERLAP_LOCAL_PAIR requires CPASYNC_PAIR=1" >&2; exit 2; }
  [[ "$DIRECTGATHER64" == 1 ]] || { echo "CPASYNC_OVERLAP_LOCAL_PAIR requires DIRECTGATHER64=1" >&2; exit 2; }
  [[ "$CPASYNC_LOCAL_PAIR" == 0 ]] || { echo "CPASYNC_OVERLAP_LOCAL_PAIR is isolated from CPASYNC_LOCAL_PAIR" >&2; exit 2; }
fi
if [[ "$CPASYNC_OVERLAP_LOCAL_PIPE2" == 1 && "$CPASYNC_OVERLAP_LOCAL_PAIR" != 1 ]]; then
  echo "CPASYNC_OVERLAP_LOCAL_PIPE2 requires CPASYNC_OVERLAP_LOCAL_PAIR=1" >&2; exit 2
fi
if [[ "$OVERLAP_LOCAL_CG" == 1 && "$CPASYNC_OVERLAP_LOCAL_PAIR" != 1 ]]; then
  echo "OVERLAP_LOCAL_CG requires CPASYNC_OVERLAP_LOCAL_PAIR=1" >&2; exit 2
fi
if [[ "$DIRECTGATHER64" == 1 ]]; then
  [[ "$FORCE7" == 0 ]] || { echo "DIRECTGATHER64 and FORCE7 are isolated modes" >&2; exit 2; }
  [[ "$PREFETCH_NEXT" == 0 ]] || { echo "DIRECTGATHER64 and PREFETCH_NEXT are isolated modes" >&2; exit 2; }
fi
if [[ "$DIRECTGATHER_SPARSE64" == 1 && "$DIRECTGATHER64" != 1 ]]; then
  echo "DIRECTGATHER_SPARSE64 requires DIRECTGATHER64=1" >&2; exit 2
fi

N="$N" OUT="$OUT" \
COL_ILP="$COL_ILP" \
DEPTHMAJOR="${DEPTHMAJOR:-1}" \
PAIR_MLP="$PAIR_MLP" \
CPASYNC_PAIR="$CPASYNC_PAIR" \
CPASYNC_LOCAL_PAIR="$CPASYNC_LOCAL_PAIR" \
CPASYNC_OVERLAP_LOCAL_PAIR="$CPASYNC_OVERLAP_LOCAL_PAIR" \
CPASYNC_OVERLAP_LOCAL_PIPE2="$CPASYNC_OVERLAP_LOCAL_PIPE2" \
OVERLAP_LOCAL_CG="$OVERLAP_LOCAL_CG" \
SORTED="$SORTED" \
PRECTX_FORWARD="$PRECTX_FORWARD" \
PRECTX_REVERSE="$PRECTX_REVERSE" \
FORCE7="$FORCE7" \
MLP_WINDOW4="$MLP_WINDOW4" \
PREFETCH_NEXT="$PREFETCH_NEXT" \
DIRECTGATHER64="$DIRECTGATHER64" \
DIRECTGATHER_SPARSE64="$DIRECTGATHER_SPARSE64" \
PM_ACCUM="${PM_ACCUM:-1}" \
MAXRREGCOUNT="${MAXRREGCOUNT:-0}" \
PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}" \
TRANSPOSE_MODE="${TRANSPOSE_MODE:-pipeline}" \
bash "$ONEESAN_ROOT/scripts/build/b300-directgather-colilp-fast.sh"

echo "b300-rankformula-hbm-max OK out=$OUT n=$N col_ilp=$COL_ILP depthmajor=${DEPTHMAJOR:-1} pair_mlp=$PAIR_MLP cpasync_pair=$CPASYNC_PAIR cpasync_local_pair=$CPASYNC_LOCAL_PAIR overlap_local_pair=$CPASYNC_OVERLAP_LOCAL_PAIR overlap_local_pipe2=$CPASYNC_OVERLAP_LOCAL_PIPE2 overlap_local_cg=$OVERLAP_LOCAL_CG directgather64=$DIRECTGATHER64 sparse64=$DIRECTGATHER_SPARSE64 sorted=$SORTED prectx_forward=$PRECTX_FORWARD prectx_reverse=$PRECTX_REVERSE window4=$MLP_WINDOW4 prefetch_next=$PREFETCH_NEXT force7=$FORCE7 maxrregcount=${MAXRREGCOUNT:-0}" >&2
