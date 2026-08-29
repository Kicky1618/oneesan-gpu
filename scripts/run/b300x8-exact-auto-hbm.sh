#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

# Production entry point for the observed low-memory-controller regime. First
# preselect the five full-pull variants with a cheap exact partial-row race, then
# full-smoke only that winner against the four warp/orbit descriptor pipelines.
# This preserves the complete-residue correctness gate while avoiding four
# redundant full n=27 forced smoke runs.
export COL_ILP="${COL_ILP:-2}"
export PAIR_MLP="${PAIR_MLP:-1}"
export RANKFORMULA_MLP_WINDOW4="${RANKFORMULA_MLP_WINDOW4:-1}"
export PM_ACCUM="${PM_ACCUM:-1}"
export CPASYNC_PAIR="${CPASYNC_PAIR:-0}"

case "$COL_ILP" in 1|2|4) ;; *) echo "COL_ILP must be 1,2,4" >&2; exit 2;; esac
for x in PAIR_MLP RANKFORMULA_MLP_WINDOW4 PM_ACCUM CPASYNC_PAIR; do
  v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0 or 1" >&2; exit 2; }
done
if [[ "$PAIR_MLP" == 1 ]]; then
  [[ "$RANKFORMULA_MLP_WINDOW4" == 1 ]] || { echo "PAIR_MLP requires WINDOW4=1" >&2; exit 2; }
  [[ "$COL_ILP" == 2 || "$COL_ILP" == 4 ]] || { echo "PAIR_MLP requires COL_ILP=2/4" >&2; exit 2; }
fi
[[ "$CPASYNC_PAIR" == 0 || "$PAIR_MLP" == 1 ]] || { echo "CPASYNC_PAIR requires PAIR_MLP=1" >&2; exit 2; }

echo "HBM staged-auto profile forced_preselect_rows=${FORCED_PRESELECT_ROWS:-1} col_ilp=$COL_ILP pair_mlp=$PAIR_MLP window4=$RANKFORMULA_MLP_WINDOW4 pm_accum=$PM_ACCUM cpasync_pair=$CPASYNC_PAIR" >&2
exec "$ONEESAN_ROOT/scripts/run/b300x8-exact-auto-hbm9-staged.sh" "$@"
