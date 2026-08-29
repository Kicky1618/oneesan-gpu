#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

helper="$ONEESAN_ROOT/src/cuda/gridfp/ramstream32_bucket_orbit_closure_pattern10_depthcode_orbitcta_flat_dynamic_pipe2_producer_warp.cuh"
pipe="$ONEESAN_ROOT/src/cuda/gridfp/ramstream32_bucket_orbit_closure_pattern10_depthcode_orbitcta_flat_dynamic_pipe2.cuh"
build="$ONEESAN_ROOT/scripts/build/b300-directgather-orbitcta-pipe2-producer-warp.sh"
ab="$ONEESAN_ROOT/scripts/bench/b300-orbitcta-flat-dynamic-pipe2-producer-warp-ab.sh"

bash -n "$build"
bash -n "$ab"
bash "$ONEESAN_ROOT/scripts/bench/b300-pipe2-producer-warp-coverage-proof.sh"
PIPE2_PRODUCER_PATCH_ONLY=1 ORBITCTA_FLAT=1 ORBITCTA_FLAT_DYNAMIC=1 ORBITCTA_FLAT_DYNAMIC_PIPE2=1 \
  ORBITCTA_FLAT_CHUNK=1 ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP=0 bash "$build"

for s in \
 'threadIdx.x < 32u' \
 'worker = uint32_t(threadIdx.x) - 32u' \
 'workers = uint32_t(blockDim.x) - 32u' \
 'group_step = workers * uint32_t(ILP)' \
 'p10dc_orbitcta_flat_forward_columns_pipe2_producer_warp' \
 'p10dc_orbitcta_flat_reverse_columns_pipe2_producer_warp'; do
 grep -Fq "$s" "$helper" || { echo "missing producer helper marker: $s" >&2; exit 3; }
done
for s in \
 'P10DC_ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_WARP' \
 'ramstream32_bucket_orbit_closure_pattern10_depthcode_orbitcta_flat_dynamic_pipe2_producer_warp.cuh' \
 'p10dc_orbitcta_flat_dynamic_pipe2_forward_columns' \
 'p10dc_orbitcta_flat_dynamic_pipe2_reverse_columns'; do
 grep -Fq "$s" "$pipe" || { echo "missing producer pipe marker: $s" >&2; exit 3; }
done

echo 'b300_pipe2_producer_warp_preflight=OK gpu_work=0 actions_triggered=0'
