#!/usr/bin/env python3
from __future__ import annotations
import pathlib,sys

if len(sys.argv)!=3:
    raise SystemExit('usage: gen-b300-main-rankstate-ilp8-cpasync-staged-wait.py INPUT.cu OUTPUT.cu')
src=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);s=src.read_text()
for req in (
    'b300_cpasync_count_ca','b300_cpasync_commit','b300_cpasync_wait_all',
    'cp.async.wait_group 0','b300_main_pull_rankstate_ilp8_kernel',
    'const Count self7=','const Count pair7=my_smem[7]',
    'const Count block7=my_smem[15]','const uint64_t mod=D_MOD;'
):
    if req not in s: raise SystemExit(f'staged cp.async wait requires artifact: {req}')
if 'b300_cpasync_wait_pair' in s or 'b300_main_rankstate_ilp8_cpasync_staged_wait=1' in s:
    raise SystemExit('source already contains staged cp.async wait')
if s.count('b300_cpasync_wait_all();') != 1:
    raise SystemExit(f'expected one ILP8 wait-all call, got {s.count("b300_cpasync_wait_all();")}')
if s.count('b300_cpasync_commit();') != 2:
    raise SystemExit(f'expected two ILP8 async groups, got {s.count("b300_cpasync_commit();")}')

# With exactly two committed groups outstanding, wait_group 1 guarantees that
# the oldest (pair-source) group is complete while the newer block-source group
# may remain in flight.  Every thread owns its own 16-Count shared slice, so no
# CTA barrier is needed before consuming the first eight slots.
anchor='''__device__ __forceinline__ void b300_cpasync_wait_all(){
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif
}
'''
if s.count(anchor)!=1:
    raise SystemExit(f'wait-all helper anchor expected one match got {s.count(anchor)}')
helper='''__device__ __forceinline__ void b300_cpasync_wait_pair(){
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 1;" ::: "memory");
#endif
}
'''
s=s.replace(anchor,helper+anchor,1)

old=[]
old.append('        b300_cpasync_wait_all();')
for k in range(8): old.append(f'        const Count pair{k}=my_smem[{k}];')
for k in range(8): old.append(f'        const Count block{k}=my_smem[{8+k}];')
old.append('        const uint64_t mod=D_MOD;')
for k in range(8):
    stmt=f'uint64_t a=uint64_t(self{k})+pair{k}+block{k};if(a>=mod)a-=mod;if(a>=mod)a-=mod;out_main[i{k}]=Count(a);'
    old.append(('        {' if k==0 else f'        if(v{k})'+'{')+stmt+'}')
old_text='\n'.join(old)
if s.count(old_text)!=1:
    raise SystemExit(f'ILP8 consume anchor expected one match got {s.count(old_text)}')

new=[]
new.append('        b300_cpasync_wait_pair();')
# Consume the completed pair group immediately.  Keep one uint64 accumulator per
# destination live across the block-group wait; this is the deliberate A/B knob
# and must be spill-gated by the benchmark/selector.
for k in range(8):
    if k==0:
        new.append(f'        uint64_t acc{k}=uint64_t(self{k})+uint64_t(my_smem[{k}]);')
    else:
        new.append(f'        uint64_t acc{k}=v{k}?uint64_t(self{k})+uint64_t(my_smem[{k}]):uint64_t(0);')
new.append('        b300_cpasync_wait_all();')
new.append('        const uint64_t mod=D_MOD;')
for k in range(8):
    body=(f'acc{k}+=uint64_t(my_smem[{8+k}]);'
          f'if(acc{k}>=mod)acc{k}-=mod;if(acc{k}>=mod)acc{k}-=mod;'
          f'out_main[i{k}]=Count(acc{k});')
    new.append(('        {' if k==0 else f'        if(v{k})'+'{')+body+'}')
new_text='\n'.join(new)
s=s.replace(old_text,new_text,1)

for req in (
    'cp.async.wait_group 1','b300_cpasync_wait_pair();',
    'uint64_t acc0=uint64_t(self0)+uint64_t(my_smem[0])',
    'acc7+=uint64_t(my_smem[15])','b300_cpasync_wait_all();'
):
    if req not in s: raise SystemExit(f'missing staged-wait artifact: {req}')
for stale in ('const Count pair7=my_smem[7]','const Count block7=my_smem[15]'):
    if stale in s: raise SystemExit(f'stale wait-all consume remains: {stale}')

out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(s)
print(f'generated {out} from {src}: b300_main_rankstate_ilp8_cpasync_staged_wait=1 wait_pair_group=1 wait_block_group=0 pair_reduce_overlap_block=1 per_thread_shared_only=1 extra_cta_barriers=0 live_u64_accumulators=8 spill_gate_required=1')
