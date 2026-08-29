#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

# PROFILED=1 consumes the exact n=21 tuner output and only races forced + the
# best warp + the best orbit profile at n=27. Keep the existing staged HBM9
# route as the default until real B300 measurements establish the profile path.
PROFILED="${PROFILED:-0}"
[[ "$PROFILED" == 0 || "$PROFILED" == 1 ]] || { echo 'PROFILED must be 0 or 1' >&2; exit 2; }
if [[ "$PROFILED" == 1 ]]; then
  PROFILE_FILE="${PROFILE_FILE:-$ONEESAN_ROOT/work/b300_hbm_profile_tune21.env}"
  [[ -f "$PROFILE_FILE" ]] || { echo "missing PROFILE_FILE=$PROFILE_FILE" >&2; echo 'run tuner first: bash scripts/bench/b300-hbm-profile-tune21.sh' >&2; exit 2; }
  export PROFILE_FILE
  echo "HBM profiled-auto profile_file=$PROFILE_FILE" >&2
  exec "$ONEESAN_ROOT/scripts/run/b300x8-exact-auto-hbm-profiled.sh" "$@"
fi

# Production entry point for the observed low-memory-controller regime. First
# preselect the full-pull variants with a cheap exact partial-row race, then
# full-smoke only that winner against the warp/orbit descriptor pipelines.
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
