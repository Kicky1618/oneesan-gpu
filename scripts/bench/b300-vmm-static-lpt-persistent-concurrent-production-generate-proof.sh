#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
SRC="$ONEESAN_ROOT/src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_fullmate_dropN.cu";OUT="${OUT:-$ONEESAN_BUILD_DIR/generated_b300_persistent_concurrent_proof.cu}"
steps=(gen-b300-vmm-production.py prune-b300-vmm-stale-shard-symbols.py lower-b300-vmm-basearg.py lower-b300-packed-group-meta.py lower-b300-staged-group-meta.py lower-b300-static-lpt-staged-meta.py lower-b300-static-lpt-staged-intervals.py lower-b300-staged-meta-kernelarg.py lower-b300-concurrent-staged-io.py lower-b300-static-worker-device-binding.py lower-b300-row-limit.py lower-b300-static-persistent-workers.py)
python3 "$ONEESAN_ROOT/scripts/build/${steps[0]}" "$SRC" "$OUT";for s in "${steps[@]:1}";do python3 "$ONEESAN_ROOT/scripts/build/$s" "$OUT" "$OUT";done
grep -Fq 'set persistent static worker device' "$OUT";grep -Fq 'main gather sync' "$OUT";grep -Fq 'block gather sync' "$OUT";grep -Fq 'main scatter sync' "$OUT";grep -Fq 'block scatter sync' "$OUT";grep -Fq 'pg.mi_stage_count,threads),threads,0,c.sMain' "$OUT";grep -Fq 'pg.di_stage_count,threads),threads,0,c.sBlock' "$OUT"
for stale in 'cudaDeviceSynchronize(),"doubleD gather sync"' 'cudaDeviceSynchronize(),"group sync"' 'cudaMemcpyToSymbol(D_GROUP_META' 'cudaMemcpy(c.dIM,pg.mi.data()' 'cudaMemcpy(c.dID,pg.di.data()';do grep -Fq "$stale" "$OUT"&&{ echo "persistent-concurrent stale=$stale" >&2;exit 3;}||true;done
echo 'b300-vmm-static-lpt-persistent-concurrent-production-generate-proof OK persistent_workers=8 concurrent_gather=1 concurrent_scatter=1 devicewide_io_sync=0 metadata_copy_per_group=0 interval_h2d_per_group=0'
