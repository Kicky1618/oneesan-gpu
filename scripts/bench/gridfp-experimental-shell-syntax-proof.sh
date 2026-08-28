#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

FILES=(
  scripts/lib/gridfp-runtime-ab-env.sh
  scripts/bench/gridfp-runtime-ab-env-proof.sh
  scripts/bench/gridfp-runtime-experimental-ab-isolation-proof.sh
  scripts/bench/gridfp-codec-table-proxy-coverage-proof.sh
  scripts/bench/gridfp-runtime-label-hotpath-proof-suite.sh
  scripts/bench/gridfp-runtime-w28-label-hotpath-suite.sh
  scripts/bench/gridfp-codec-table-layout-suite.sh
  scripts/bench/gridfp-codec-table-exact-suite.sh
  scripts/bench/gridfp-build-nvcc-prepend-smoke.sh
  scripts/bench/gridfp-reduced-runtime-owner-label-hotpath-ab.sh
  scripts/bench/gridfp-reduced-runtime-owner-local-sector-compact-table-ab.sh
  scripts/bench/gridfp-reduced-runtime-label-hotpath-cumulative-ab.sh
  scripts/bench/gridfp-reduced-runtime-materialize-primitive-last-r-ab.sh
  scripts/bench/gridfp-reduced-runtime-materialize-primitive-packed-ab.sh
  scripts/bench/gridfp-reduced-runtime-primitive1-u32-ab.sh
  scripts/bench/gridfp-reduced-runtime-primitive-sym-u32-ab.sh
  scripts/bench/gridfp-reduced-runtime-choose-sym-u32-ab.sh
  scripts/bench/gridfp-reduced-runtime-codec-tables-sym-u32-ab.sh
  scripts/bench/gridfp-primitive1-u32-table-microprobe.sh
  scripts/bench/gridfp-runtime-primitive-rank-packed-microprobe.sh
  scripts/bench/gridfp-primitive-sym-u32-table-microprobe.sh
  scripts/bench/gridfp-choose-sym-u32-table-microprobe.sh
  scripts/bench/gridfp-motzkin-tri-u64-table-microprobe.sh
  scripts/bench/gridfp-primitive1-u32-table-proof.sh
  scripts/bench/gridfp-runtime-primitive-rank-packed-proof.sh
  scripts/bench/gridfp-primitive-sym-u32-table-proof.sh
  scripts/bench/gridfp-choose-sym-u32-table-proof.sh
  scripts/bench/gridfp-codec-table-budget-proof.sh
)

checked=0
for rel in "${FILES[@]}"; do
  path="$ONEESAN_ROOT/$rel"
  [[ -f "$path" ]] || { echo "missing syntax target: $rel" >&2; exit 2; }
  bash -n "$path"
  ((checked += 1))
done

echo "gridfp-experimental-shell-syntax-proof OK files=$checked exact=1"
