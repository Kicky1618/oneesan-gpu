#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

files=(
  "$ONEESAN_ROOT/scripts/build/b300-directgather-orbitcta-pipe2-producer-warp.sh"
  "$ONEESAN_ROOT/scripts/bench/b300-pipe2-producer-warp-coverage-proof.sh"
  "$ONEESAN_ROOT/scripts/bench/b300-orbitcta-flat-dynamic-pipe2-producer-adaptive-weight-ab.sh"
  "$ONEESAN_ROOT/scripts/bench/b300-hbm-profile-refine-orbit-dynamic-producer-adaptive21.sh"
  "$ONEESAN_ROOT/scripts/bench/b300-hbm-profile-auto21-producer-adaptive.sh"
  "$ONEESAN_ROOT/scripts/run/b300x8-exact-auto-hbm-profiled-producer-adaptive.sh"
)
for f in "${files[@]}"; do
  [[ -f "$f" ]] || { echo "missing $f" >&2; exit 2; }
  bash -n "$f"
done

cov="$(bash "$ONEESAN_ROOT/scripts/bench/b300-pipe2-producer-warp-coverage-proof.sh")"
grep -q 'b300_pipe2_producer_warp_coverage=OK' <<<"$cov" || exit 3
grep -q 'producer_adaptive_cols=.*wide_weight=1 exact_once=1' <<<"$cov" || exit 3

patch="$(PIPE2_PRODUCER_PATCH_ONLY=1 PRODUCER_WORKER_WEIGHT=4 PRODUCER_ADAPTIVE_COLS=4096 QUAD_MLP=1 ORBITCTA_COL_ILP=4 PRECTX_FORWARD=1 PRECTX_REVERSE=1 PRECTX_COMPACT=1 bash "$ONEESAN_ROOT/scripts/build/b300-directgather-orbitcta-pipe2-producer-warp.sh")"
grep -q 'b300_pipe2_producer_warp_patch=OK' <<<"$patch" || exit 4
grep -q 'producer_worker_weight=4' <<<"$patch" || exit 4
grep -q 'producer_adaptive_cols=4096' <<<"$patch" || exit 4

echo 'b300_pipe2_producer_adaptive_source_preflight=OK gpu_work=0'
