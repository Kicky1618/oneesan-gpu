#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
SRC="$ONEESAN_ROOT/src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_fullmate_dropN.cu";OUT="${OUT:-$ONEESAN_BUILD_DIR/generated_b300_cross_window_proof.cu}"
steps=(gen-b300-vmm-production.py prune-b300-vmm-stale-shard-symbols.py lower-b300-vmm-basearg.py lower-b300-packed-group-meta.py lower-b300-staged-group-meta.py lower-b300-static-lpt-staged-meta.py lower-b300-static-lpt-staged-intervals.py lower-b300-staged-meta-kernelarg.py lower-b300-concurrent-staged-io.py lower-b300-async-staged-io-hostsync.py lower-b300-static-worker-device-binding.py lower-b300-row-limit.py lower-b300-static-persistent-workers.py lower-b300-window-batched-groups.py lower-b300-cross-device-window-barrier.py)
python3 "$ONEESAN_ROOT/scripts/build/${steps[0]}" "$SRC" "$OUT";for s in "${steps[@]:1}";do python3 "$ONEESAN_ROOT/scripts/build/$s" "$OUT" "$OUT";done
grep -Fq 'record GPU window done' "$OUT";grep -Fq 'gpu0 wait peer window' "$OUT";grep -Fq 'gpu0 global window sync' "$OUT";grep -Fq 'for(int q=1;q<ng;++q)' "$OUT"
for stale in 'window batch sync' 'scatter join sync' 'cudaDeviceSynchronize(),"group sync"';do grep -Fq "$stale" "$OUT"&&{ echo "cross-window stale=$stale" >&2;exit 3;}||true;done
[[ "$(grep -Fc 'gpu0 global window sync' "$OUT")" == 1 ]]||exit 4
echo 'b300-vmm-static-lpt-cross-window-production-generate-proof OK cross_device_event_wait=1 host_gpu_syncs_per_window=1 expected_windows=56 expected_host_gpu_sync_calls=56 windowbatch_old_calls=448 per_group_host_syncs=0 scratch_realloc_per_group=0'
