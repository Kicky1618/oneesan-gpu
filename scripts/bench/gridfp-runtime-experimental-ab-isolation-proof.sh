#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

FILES=(
  scripts/bench/gridfp-reduced-runtime-owner-label-hotpath-ab.sh
  scripts/bench/gridfp-reduced-runtime-owner-local-sector-compact-table-ab.sh
  scripts/bench/gridfp-reduced-runtime-label-hotpath-cumulative-ab.sh
  scripts/bench/gridfp-reduced-runtime-materialize-primitive-last-r-ab.sh
  scripts/bench/gridfp-reduced-runtime-materialize-primitive-packed-ab.sh
  scripts/bench/gridfp-reduced-runtime-primitive1-u32-ab.sh
  scripts/bench/gridfp-reduced-runtime-primitive-sym-u32-ab.sh
  scripts/bench/gridfp-reduced-runtime-choose-sym-u32-ab.sh
  scripts/bench/gridfp-reduced-runtime-codec-tables-sym-u32-ab.sh
)

checked=0
for rel in "${FILES[@]}"; do
  path="$ONEESAN_ROOT/$rel"
  [[ -f "$path" ]] || { echo "missing isolated A/B: $rel" >&2; exit 2; }
  grep -Fq 'gridfp-runtime-ab-env.sh' "$path" || {
    echo "isolated A/B does not source canonical env: $rel" >&2; exit 3; }
  grep -Fq 'env "${GRIDFP_RUNTIME_AB_ENV[@]}"' "$path" || {
    echo "isolated A/B does not pass canonical env: $rel" >&2; exit 4; }
  grep -Fq 'gridfp_runtime_ab_assert_build' "$path" || {
    echo "isolated A/B does not verify build stack: $rel" >&2; exit 5; }
  grep -Fq 'MODE=two-row-runtime-multigpu' "$path" || {
    echo "isolated A/B is not a two-row runtime experiment: $rel" >&2; exit 6; }
  grep -Fq 'NVCC_PREPEND_FLAGS=' "$path" || {
    echo "isolated A/B has no isolated experimental injection: $rel" >&2; exit 7; }

  # Canonical runtime knobs belong in gridfp-runtime-ab-env.sh. A local
  # RUNTIME_*= assignment would silently override/drift the shared baseline.
  if grep -Eq '(^|[[:space:]])RUNTIME_[A-Z0-9_]+=' "$path"; then
    echo "isolated A/B contains local runtime knob assignment: $rel" >&2
    grep -En '(^|[[:space:]])RUNTIME_[A-Z0-9_]+=' "$path" >&2 || true
    exit 8
  fi
  bash -n "$path"
  ((checked += 1))
done

[[ "$checked" == 9 ]] || { echo "unexpected isolated A/B count: $checked" >&2; exit 9; }
echo "gridfp-runtime-experimental-ab-isolation-proof OK files=$checked canonical_env=1 build_assert=1 no_local_runtime_knobs=1 exact=1"
