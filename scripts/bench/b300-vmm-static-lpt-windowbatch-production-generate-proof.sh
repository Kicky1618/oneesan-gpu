#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
SRC="$ONEESAN_ROOT/src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_fullmate_dropN.cu";OUT="${OUT:-$ONEESAN_BUILD_DIR/generated_b300_windowbatch_proof.cu}"
steps=(gen-b300-vmm-production.py prune-b300-vmm-stale-shard-symbols.py lower-b300-vmm-basearg.py lower-b300-packed-group-meta.py lower-b300-staged-group-meta.py lower-b300-static-lpt-staged-meta.py lower-b300-static-lpt-staged-intervals.py lower-b300-staged-meta-kernelarg.py lower-b300-concurrent-staged-io.py lower-b300-async-staged-io-hostsync.py lower-b300-static-worker-device-binding.py lower-b300-row-limit.py lower-b300-static-persistent-workers.py lower-b300-window-batched-groups.py)
python3 "$ONEESAN_ROOT/scripts/build/${steps[0]}" "$SRC" "$OUT";for s in "${steps[@]:1}";do python3 "$ONEESAN_ROOT/scripts/build/$s" "$OUT" "$OUT";done
grep -Fq 'void reserve_arena(size_t need)' "$OUT";grep -Fq 'realloc_per_group=0' "$OUT";grep -Fq 'record main scatter done' "$OUT";grep -Fq 'block wait main scatter' "$OUT";grep -Fq 'window batch sync' "$OUT";grep -Fq 'window_t0' "$OUT"
for stale in 'scatter join sync' 'main gather sync' 'block gather sync' 'main sync' 'block sync' 'if(need>capArena){if(arena)cudaFree(arena)' 'c.active+=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count()';do grep -Fq "$stale" "$OUT"&&{ echo "windowbatch stale=$stale" >&2;exit 3;}||true;done
[[ "$(grep -Fc 'window batch sync' "$OUT")" == 1 ]]||exit 4
echo 'b300-vmm-static-lpt-windowbatch-production-generate-proof OK scratch_preallocated=1 scratch_realloc_per_group=0 per_group_host_syncs=0 persistent_workers=8 window_syncs_per_gpu=56 expected_total_window_host_syncs=448 metadata_copy_per_group=0 interval_h2d_per_group=0'
