#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
SRC="$ONEESAN_ROOT/src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_fullmate_dropN.cu"
GEN="$ONEESAN_ROOT/scripts/build/gen-b300-vmm-production.py"
PRUNE="$ONEESAN_ROOT/scripts/build/prune-b300-vmm-stale-shard-symbols.py"
BASEARG="$ONEESAN_ROOT/scripts/build/lower-b300-vmm-basearg.py"
PACK="$ONEESAN_ROOT/scripts/build/lower-b300-packed-group-meta.py"
STAGE="$ONEESAN_ROOT/scripts/build/lower-b300-staged-group-meta.py"
STATIC="$ONEESAN_ROOT/scripts/build/lower-b300-static-lpt-staged-meta.py"
INTERVALS="$ONEESAN_ROOT/scripts/build/lower-b300-static-lpt-staged-intervals.py"
METAARG="$ONEESAN_ROOT/scripts/build/lower-b300-staged-meta-kernelarg.py"
BIND="$ONEESAN_ROOT/scripts/build/lower-b300-static-worker-device-binding.py"
ROWLIMIT="$ONEESAN_ROOT/scripts/build/lower-b300-row-limit.py"
OUT="${OUT:-$ONEESAN_BUILD_DIR/generated_b300_static_lpt_control_bundle_proof.cu}"

bash "$ONEESAN_ROOT/scripts/bench/b300-staged-group-meta-plan-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/b300-static-lpt-local-meta-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/b300-static-lpt-interval-staging-proof.sh"
python3 "$GEN" "$SRC" "$OUT"
python3 "$PRUNE" "$OUT" "$OUT"
python3 "$BASEARG" "$OUT" "$OUT"
python3 "$PACK" "$OUT" "$OUT"
python3 "$STAGE" "$OUT" "$OUT"
python3 "$STATIC" "$OUT" "$OUT"
python3 "$INTERVALS" "$OUT" "$OUT"
python3 "$METAARG" "$OUT" "$OUT"
python3 "$BIND" "$OUT" "$OUT"
python3 "$ROWLIMIT" "$OUT" "$OUT"

grep -Fq 'scheduler=static_lpt' "$OUT"
grep -Fq 'PeerInterval*dStageIntervals=nullptr' "$OUT"
grep -Fq 'copy_mode=H2D_once_local_then_zero_interval_copy_per_group' "$OUT"
grep -Fq 'const DeviceGroupMeta*group_meta=c.dGroupMeta+pg.meta_id' "$OUT"
grep -Fq 'meta_access=staged_global_kernelarg' "$OUT"
grep -Fq 'cudaSetDevice(d),"set static worker device"' "$OUT"
grep -Fq 'for(int row=0;row<b300_row_limit;++row)' "$OUT"
grep -Fq 'combined_stage_max_bytes!=53961480ull' "$OUT"

for stale in \
  'std::atomic<int>next{0}' \
  'next.fetch_add(1,std::memory_order_relaxed)' \
  'cudaMemcpy(c.dIM,pg.mi.data()' \
  'cudaMemcpy(c.dID,pg.di.data()' \
  'cudaMemcpyToSymbol(D_GROUP_META' \
  'staged group meta D2D' \
  'D_GROUP_META' \
  'cudaSetDevice(c.dev),"set worker"' \
  'cudaSetDevice(dev),"set ensure"'; do
  if grep -Fq "$stale" "$OUT"; then
    echo "control-bundle stale artifact remains: $stale" >&2
    exit 3
  fi
done

[[ "$(grep -Fc 'cudaSetDevice(d),"set static worker device"' "$OUT")" == 1 ]] || { echo "worker binding count mismatch" >&2; exit 4; }

echo "b300-vmm-static-lpt-control-bundle-production-generate-proof OK scheduler=static_lpt metadata_replication=0 per_group_meta_copy_bytes=0 per_group_interval_h2d=0 static_worker_device_binding=1 expected_default_combined_stage_max_mib_per_gpu=51.461677551 expected_old_worker_cudaSetDevice_calls=917504 expected_new_worker_cudaSetDevice_calls=448 row_limit_default_full=1"
