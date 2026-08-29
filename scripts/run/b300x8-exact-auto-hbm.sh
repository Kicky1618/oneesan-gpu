#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

# Production entry point for the observed low-memory-controller regime. Race the
# current forced2 full-pull backend against both warp-striped and orbit-CTA pair64
# memory pipelines, in dense and sparse descriptor encodings. The HBM5 selector
# exact-compares one CRT residue, records wall/MC utilization, keeps the winning
# smoke result as a checkpoint, then continues only the fastest correct backend.
export COL_ILP="${COL_ILP:-2}"
export PAIR_MLP="${PAIR_MLP:-1}"
export RANKFORMULA_MLP_WINDOW4="${RANKFORMULA_MLP_WINDOW4:-1}"
export PM_ACCUM="${PM_ACCUM:-1}"
export CPASYNC_PAIR="${CPASYNC_PAIR:-0}"
export CANDIDATES="${CANDIDATES:-forced warp_dense warp_sparse orbit_dense orbit_sparse}"

case "$COL_ILP" in 1|2|4) ;; *) echo "COL_ILP must be 1,2,4" >&2; exit 2;; esac
for x in PAIR_MLP RANKFORMULA_MLP_WINDOW4 PM_ACCUM CPASYNC_PAIR; do
  v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0 or 1" >&2; exit 2; }
done
if [[ "$PAIR_MLP" == 1 ]]; then
  [[ "$RANKFORMULA_MLP_WINDOW4" == 1 ]] || { echo "PAIR_MLP requires WINDOW4=1" >&2; exit 2; }
  [[ "$COL_ILP" == 2 || "$COL_ILP" == 4 ]] || { echo "PAIR_MLP requires COL_ILP=2/4" >&2; exit 2; }
fi
[[ "$CPASYNC_PAIR" == 0 || "$PAIR_MLP" == 1 ]] || { echo "CPASYNC_PAIR requires PAIR_MLP=1" >&2; exit 2; }

echo "HBM auto5 profile col_ilp=$COL_ILP pair_mlp=$PAIR_MLP window4=$RANKFORMULA_MLP_WINDOW4 pm_accum=$PM_ACCUM cpasync_pair=$CPASYNC_PAIR candidates='$CANDIDATES'" >&2
exec "$ONEESAN_ROOT/scripts/run/b300x8-exact-auto-hbm5.sh" "$@"
