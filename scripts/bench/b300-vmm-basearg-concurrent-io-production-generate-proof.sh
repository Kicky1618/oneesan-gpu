#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
SRC="$ONEESAN_ROOT/src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_fullmate_dropN.cu"
GEN="$ONEESAN_ROOT/scripts/build/gen-b300-vmm-production.py";PRUNE="$ONEESAN_ROOT/scripts/build/prune-b300-vmm-stale-shard-symbols.py";BASEARG="$ONEESAN_ROOT/scripts/build/lower-b300-vmm-basearg.py";CONCURRENT="$ONEESAN_ROOT/scripts/build/lower-b300-concurrent-io.py"
OUT="${OUT:-$ONEESAN_BUILD_DIR/generated_b300_vmm_basearg_concurrent_io_proof.cu}"
bash "$ONEESAN_ROOT/scripts/bench/b300-vmm-basearg-production-generate-proof.sh"
python3 "$GEN" "$SRC" "$OUT";python3 "$PRUNE" "$OUT" "$OUT";python3 "$BASEARG" "$OUT" "$OUT";python3 "$CONCURRENT" "$OUT" "$OUT"
grep -Fq 'interval_io_kernel<false><<<interval_blocks(pg.mi.size(),threads),threads,0,c.sMain>>>' "$OUT"
grep -Fq 'interval_io_kernel<false><<<interval_blocks(pg.di.size(),threads),threads,0,c.sBlock>>>' "$OUT"
grep -Fq 'interval_io_kernel<true><<<interval_blocks(pg.mi.size(),threads),threads,0,c.sMain>>>' "$OUT"
grep -Fq 'interval_io_kernel<true><<<interval_blocks(pg.di.size(),threads),threads,0,c.sBlock>>>' "$OUT"
grep -Fq 'cudaStreamSynchronize(c.sMain),"main gather sync"' "$OUT"
grep -Fq 'cudaStreamSynchronize(c.sBlock),"block gather sync"' "$OUT"
grep -Fq 'cudaStreamSynchronize(c.sMain),"main scatter sync"' "$OUT"
grep -Fq 'cudaStreamSynchronize(c.sBlock),"block scatter sync"' "$OUT"
for stale in 'cudaDeviceSynchronize(),"doubleD gather sync"' 'cudaDeviceSynchronize(),"group sync"';do if grep -Fq "$stale" "$OUT";then echo "concurrent I/O source still contains device-wide sync: $stale" >&2;exit 3;fi;done
echo "b300-vmm-basearg-concurrent-io-production-generate-proof OK gather_streams=2 scatter_streams=2 devicewide_io_sync=0 expected_default_overlap_groups=16384 expected_default_overlap_ratio=1.0"
