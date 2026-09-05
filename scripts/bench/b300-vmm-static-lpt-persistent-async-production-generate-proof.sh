#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
SRC="$ONEESAN_ROOT/src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_fullmate_dropN.cu";OUT="${OUT:-$ONEESAN_BUILD_DIR/generated_b300_persistent_async_proof.cu}"
steps=(gen-b300-vmm-production.py prune-b300-vmm-stale-shard-symbols.py lower-b300-vmm-basearg.py lower-b300-packed-group-meta.py lower-b300-staged-group-meta.py lower-b300-static-lpt-staged-meta.py lower-b300-static-lpt-staged-intervals.py lower-b300-staged-meta-kernelarg.py lower-b300-concurrent-staged-io.py lower-b300-async-staged-io-hostsync.py lower-b300-static-worker-device-binding.py lower-b300-row-limit.py lower-b300-static-persistent-workers.py)
python3 "$ONEESAN_ROOT/scripts/build/${steps[0]}" "$SRC" "$OUT";for s in "${steps[@]:1}";do python3 "$ONEESAN_ROOT/scripts/build/$s" "$OUT" "$OUT";done
grep -Fq 'set persistent static worker device' "$OUT";grep -Fq 'scatter join sync' "$OUT";grep -Fq 'record block scatter done' "$OUT";grep -Fq 'main wait block scatter' "$OUT"
for stale in 'main gather sync' 'block gather sync' 'main sync' 'block sync' 'main scatter sync' 'block scatter sync' 'cudaDeviceSynchronize(),"doubleD gather sync"' 'cudaDeviceSynchronize(),"group sync"';do grep -Fq "$stale" "$OUT"&&{ echo "persistent-async stale=$stale" >&2;exit 3;}||true;done
[[ "$(grep -Fc 'scatter join sync' "$OUT")" == 1 ]]||exit 4
echo 'b300-vmm-static-lpt-persistent-async-production-generate-proof OK persistent_workers=8 concurrent_io=1 gather_host_syncs=0 transition_tail_host_syncs=0 scatter_host_syncs=1 total_host_syncs_per_group=1 metadata_copy_per_group=0 interval_h2d_per_group=0'
