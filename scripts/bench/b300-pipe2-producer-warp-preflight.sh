#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

helper="$ONEESAN_ROOT/src/cuda/gridfp/ramstream32_bucket_orbit_closure_pattern10_depthcode_orbitcta_flat_dynamic_pipe2_producer_warp.cuh"
pipe="$ONEESAN_ROOT/src/cuda/gridfp/ramstream32_bucket_orbit_closure_pattern10_depthcode_orbitcta_flat_dynamic_pipe2.cuh"
build="$ONEESAN_ROOT/scripts/build/b300-directgather-orbitcta-pipe2-producer-warp.sh"
old_ab="$ONEESAN_ROOT/scripts/bench/b300-orbitcta-flat-dynamic-pipe2-producer-warp-ab.sh"
selected_ab="$ONEESAN_ROOT/scripts/bench/b300-orbitcta-flat-dynamic-pipe2-producer-selected-ab.sh"
prectx_ab="$ONEESAN_ROOT/scripts/bench/b300-orbitcta-flat-dynamic-pipe2-producer-prectx-warpcoop-ab.sh"
producer_refine="$ONEESAN_ROOT/scripts/bench/b300-hbm-profile-refine-orbit-dynamic-producer21.sh"
producer_prectx_refine="$ONEESAN_ROOT/scripts/bench/b300-hbm-profile-refine-orbit-dynamic-producer-prectx21.sh"
auto="$ONEESAN_ROOT/scripts/bench/b300-hbm-profile-auto21.sh"
run="$ONEESAN_ROOT/scripts/run/b300x8-exact-auto-hbm-profiled.sh"
staged="$ONEESAN_ROOT/scripts/run/b300x8-exact-auto-hbm-profiled-staged.sh"

tmp_basic="${TMPDIR:-/tmp}/oneesan-producer-patch-basic.$$.out"
tmp_weighted="${TMPDIR:-/tmp}/oneesan-producer-patch-weighted.$$.out"
tmp_prectx="${TMPDIR:-/tmp}/oneesan-producer-patch-prectx.$$.out"
trap 'rm -f "$tmp_basic" "$tmp_weighted" "$tmp_prectx"' EXIT

for s in "$build" "$old_ab" "$selected_ab" "$prectx_ab" "$producer_refine" "$producer_prectx_refine" "$auto" "$run" "$staged"; do
  bash -n "$s"
done
bash "$ONEESAN_ROOT/scripts/bench/b300-pipe2-producer-warp-coverage-proof.sh"

PIPE2_PRODUCER_PATCH_ONLY=1 ORBITCTA_FLAT=1 ORBITCTA_FLAT_DYNAMIC=1 ORBITCTA_FLAT_DYNAMIC_PIPE2=1 \
  ORBITCTA_FLAT_CHUNK=1 ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP=0 bash "$build" >"$tmp_basic"
grep -q 'b300_pipe2_producer_warp_patch=OK' "$tmp_basic"
grep -q 'producer_worker_weight=0' "$tmp_basic"

PIPE2_PRODUCER_PATCH_ONLY=1 PRODUCER_WORKER_WEIGHT=4 ORBITCTA_FLAT=1 ORBITCTA_FLAT_DYNAMIC=1 ORBITCTA_FLAT_DYNAMIC_PIPE2=1 \
  ORBITCTA_FLAT_CHUNK=1 ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP=0 bash "$build" >"$tmp_weighted"
grep -q 'producer_worker_weight=4' "$tmp_weighted"

PIPE2_PRODUCER_PATCH_ONLY=1 PRODUCER_PRECTX_WARPCOOP=1 QUAD_MLP=1 ORBITCTA_COL_ILP=4 \
  PRECTX_FORWARD=1 PRECTX_REVERSE=1 PRECTX_COMPACT=1 PRECTX_FLAT_BID=1 PRECTX_FLAT_BID_FUSED=0 \
  ORBITCTA_FLAT=1 ORBITCTA_FLAT_DYNAMIC=1 ORBITCTA_FLAT_DYNAMIC_PIPE2=1 ORBITCTA_FLAT_CHUNK=1 ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP=0 \
  bash "$build" >"$tmp_prectx"
grep -q 'producer_prectx_warpcoop=1 prectx_flat_bid=1 prectx_flat_bid_fused=0' "$tmp_prectx"

for s in \
 'P10DC_ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_WORKER_WEIGHT' \
 'p10dc_orbitcta_flat_pipe2_producer_partition' \
 'first_slot' \
 'slot_count' \
 'logical_workers' \
 'group_step = workers * uint32_t(ILP)' \
 'p10dc_orbitcta_flat_forward_columns_pipe2_producer_warp' \
 'p10dc_orbitcta_flat_reverse_columns_pipe2_producer_warp'; do
  grep -Fq "$s" "$helper" || { echo "missing producer helper marker: $s" >&2; exit 3; }
done
for s in \
 'P10DC_ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_WARP' \
 'P10DC_ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_PRECTX_WARPCOOP' \
 'ramstream32_bucket_orbit_closure_pattern10_depthcode_orbitcta_flat_dynamic_pipe2_producer_warp.cuh' \
 'p10dc_orbitcta_flat_dynamic_pipe2_forward_columns' \
 'p10dc_orbitcta_flat_dynamic_pipe2_reverse_columns'; do
  grep -Fq "$s" "$pipe" || { echo "missing producer pipe marker: $s" >&2; exit 3; }
done
for s in \
 'DYNAMIC_PRODUCER_PROFILE=' \
 'DYNAMIC_PRODUCER_PRECTX_PROFILE=' \
 'ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_WORKER_WEIGHT' \
 'b300-hbm-profile-refine-orbit-dynamic-producer21.sh' \
 'b300-hbm-profile-refine-orbit-dynamic-producer-prectx21.sh'; do
  grep -Fq "$s" "$auto" || { echo "missing auto producer marker: $s" >&2; exit 3; }
done
for s in \
 'ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_WARP=' \
 'ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_WORKER_WEIGHT=' \
 'ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_PRECTX_WARPCOOP=' \
 '_dpw${ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_WARP}_pww${ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_WORKER_WEIGHT}_ppw${ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_PRECTX_WARPCOOP}' \
 'b300-directgather-orbitcta-pipe2-producer-warp.sh'; do
  grep -Fq "$s" "$run" || { echo "missing n27 producer marker: $s" >&2; exit 3; }
done
for s in \
 'ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_WORKER_WEIGHT=' \
 '_dpw${ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_WARP}_pww${ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_WORKER_WEIGHT}_ppw${ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_PRECTX_WARPCOOP}'; do
  grep -Fq "$s" "$staged" || { echo "missing staged weighted producer marker: $s" >&2; exit 3; }
done

echo 'b300_pipe2_producer_warp_preflight=OK producer_profile=OK weighted_producer=OK producer_prectx=OK n27=OK gpu_work=0 actions_triggered=0'
