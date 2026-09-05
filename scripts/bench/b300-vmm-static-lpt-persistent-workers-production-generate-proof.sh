#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
SRC="$ONEESAN_ROOT/src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_fullmate_dropN.cu";OUT="${OUT:-$ONEESAN_BUILD_DIR/generated_b300_persistent_workers_proof.cu}"
steps=(gen-b300-vmm-production.py prune-b300-vmm-stale-shard-symbols.py lower-b300-vmm-basearg.py lower-b300-packed-group-meta.py lower-b300-staged-group-meta.py lower-b300-static-lpt-staged-meta.py lower-b300-static-lpt-staged-intervals.py lower-b300-staged-meta-kernelarg.py lower-b300-static-worker-device-binding.py lower-b300-row-limit.py lower-b300-static-persistent-workers.py)
python3 "$ONEESAN_ROOT/scripts/build/${steps[0]}" "$SRC" "$OUT"
for s in "${steps[@]:1}";do python3 "$ONEESAN_ROOT/scripts/build/$s" "$OUT" "$OUT";done
grep -Fq 'struct B300HostBarrier' "$OUT";grep -Fq 'host_barrier.wait()' "$OUT";grep -Fq 'set persistent static worker device' "$OUT";grep -Fq 'for(size_t wi=0;wi<schedule.size();++wi)' "$OUT";grep -Fq 'std::atomic<int>done_windows_atomic' "$OUT"
for stale in 'set static worker device' 'set worker' 'set ensure' 'std::atomic<int>next{0}' 'cudaMemcpyToSymbol(D_GROUP_META' 'cudaMemcpy(c.dIM,pg.mi.data()' 'cudaMemcpy(c.dID,pg.di.data()';do grep -Fq "$stale" "$OUT"&&{ echo "persistent generated source stale=$stale" >&2;exit 3;}||true;done
[[ "$(grep -Fc 'set persistent static worker device' "$OUT")" == 1 ]]||exit 4
echo 'b300-vmm-static-lpt-persistent-workers-production-generate-proof OK persistent_workers=8 window_barrier=1 thread_creations_old=448 thread_creations_new=8 cudaSetDevice_old=917504 cudaSetDevice_new=8 meta_copy_per_group=0 interval_h2d_per_group=0'
