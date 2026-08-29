#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

# Aggressive B300 latency-hiding preset for the observed low memory-controller
# utilization regime. Depth-major makes descriptor reads contiguous. Pair MLP
# issues useful source reads for two columns before reduction while WINDOW4
# bounds register pressure. CPASYNC_PAIR and SORTED are opt-in experiments until
# their B300 exact A/B measurements show a win.
N="${N:-27}"
COL_ILP="${COL_ILP:-2}"
PAIR_MLP="${PAIR_MLP:-1}"
CPASYNC_PAIR="${CPASYNC_PAIR:-0}"
SORTED="${SORTED:-0}"
FORCE7="${FORCE7:-0}"
MLP_WINDOW4="${MLP_WINDOW4:-1}"
PREFETCH_NEXT="${PREFETCH_NEXT:-0}"
OUT="${OUT:-$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_rankformula_hbm_max_n${N}}"

if [[ "$PAIR_MLP" == 1 ]]; then
  [[ "$COL_ILP" == 2 || "$COL_ILP" == 4 ]] || { echo "PAIR_MLP requires COL_ILP=2 or 4" >&2; exit 2; }
  [[ "$MLP_WINDOW4" == 1 ]] || { echo "PAIR_MLP requires MLP_WINDOW4=1" >&2; exit 2; }
  [[ "$FORCE7" == 0 ]] || { echo "PAIR_MLP and FORCE7 are isolated modes" >&2; exit 2; }
fi
if [[ "$CPASYNC_PAIR" == 1 ]]; then
  [[ "$PAIR_MLP" == 1 ]] || { echo "CPASYNC_PAIR requires PAIR_MLP=1" >&2; exit 2; }
  [[ "$PREFETCH_NEXT" == 0 ]] || { echo "CPASYNC_PAIR and PREFETCH_NEXT are isolated modes" >&2; exit 2; }
fi

N="$N" OUT="$OUT" \
COL_ILP="$COL_ILP" \
DEPTHMAJOR="${DEPTHMAJOR:-1}" \
PAIR_MLP="$PAIR_MLP" \
CPASYNC_PAIR="$CPASYNC_PAIR" \
SORTED="$SORTED" \
FORCE7="$FORCE7" \
MLP_WINDOW4="$MLP_WINDOW4" \
PREFETCH_NEXT="$PREFETCH_NEXT" \
PM_ACCUM="${PM_ACCUM:-1}" \
MAXRREGCOUNT="${MAXRREGCOUNT:-0}" \
PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}" \
TRANSPOSE_MODE="${TRANSPOSE_MODE:-pipeline}" \
bash "$ONEESAN_ROOT/scripts/build/b300-directgather-colilp-fast.sh"

echo "b300-rankformula-hbm-max OK out=$OUT n=$N col_ilp=$COL_ILP depthmajor=${DEPTHMAJOR:-1} pair_mlp=$PAIR_MLP cpasync_pair=$CPASYNC_PAIR sorted=$SORTED window4=$MLP_WINDOW4 prefetch_next=$PREFETCH_NEXT force7=$FORCE7 maxrregcount=${MAXRREGCOUNT:-0}" >&2
