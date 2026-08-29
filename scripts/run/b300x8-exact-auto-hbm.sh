#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

# Compile orbit candidates with the low-memory-controller latency-hiding stack.
# b300x8-exact-auto.sh intentionally owns correctness gating, residue matching,
# timing, MC sampling, backend selection and checkpoint seeding. These variables
# are exported so its child orbit build inherits the same compile-time profile.
export ORBITCTA_COL_ILP="${ORBITCTA_COL_ILP:-2}"
export PAIR_MLP="${PAIR_MLP:-1}"
export RANKFORMULA_MLP_WINDOW4="${RANKFORMULA_MLP_WINDOW4:-1}"
export PM_ACCUM="${PM_ACCUM:-1}"
# cp.async remains opt-in until the remote-peer microprobe and actual B300 A/B
# show a win. The register pair path already exposes two-column source MLP.
export CPASYNC_PAIR="${CPASYNC_PAIR:-0}"
export CANDIDATES="${CANDIDATES:-forced orbit_dense orbit_sparse}"

case "$ORBITCTA_COL_ILP" in 1|2|4) ;; *) echo "ORBITCTA_COL_ILP must be 1,2,4" >&2; exit 2;; esac
for x in PAIR_MLP RANKFORMULA_MLP_WINDOW4 PM_ACCUM CPASYNC_PAIR; do
  v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0 or 1" >&2; exit 2; }
done
if [[ "$PAIR_MLP" == 1 ]]; then
  [[ "$RANKFORMULA_MLP_WINDOW4" == 1 ]] || { echo "PAIR_MLP requires WINDOW4=1" >&2; exit 2; }
  [[ "$ORBITCTA_COL_ILP" == 2 || "$ORBITCTA_COL_ILP" == 4 ]] || { echo "PAIR_MLP requires ILP2/4" >&2; exit 2; }
fi
[[ "$CPASYNC_PAIR" == 0 || "$PAIR_MLP" == 1 ]] || { echo "CPASYNC_PAIR requires PAIR_MLP=1" >&2; exit 2; }

echo "HBM auto profile orbit_ilp=$ORBITCTA_COL_ILP pair_mlp=$PAIR_MLP window4=$RANKFORMULA_MLP_WINDOW4 pm_accum=$PM_ACCUM cpasync_pair=$CPASYNC_PAIR candidates='$CANDIDATES'" >&2
exec "$ONEESAN_ROOT/scripts/run/b300x8-exact-auto.sh" "$@"
