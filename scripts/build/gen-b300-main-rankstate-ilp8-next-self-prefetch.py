#!/usr/bin/env python3
from __future__ import annotations
import pathlib,sys

if len(sys.argv)!=3:
    raise SystemExit('usage: gen-b300-main-rankstate-ilp8-next-self-prefetch.py INPUT.cu OUTPUT.cu')
src=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);s=src.read_text()
for req in ('b300_main_pull_rankstate_ilp8_kernel','base+=8*grid','const Count self7='):
    if req not in s: raise SystemExit(f'next-self prefetch requires artifact: {req}')
if 'b300_prefetch_next_self_l2' in s: raise SystemExit('source already has next-self prefetch')
marker='\n\nstatic Code rank_full(MateID m,int width)'
if marker not in s: raise SystemExit('rank_full marker not found')
helper=r'''

__device__ __forceinline__ void b300_prefetch_next_self_l2(const Count* p,bool valid){
#if __CUDA_ARCH__ >= 700
    if(valid){
        const unsigned long long a=reinterpret_cast<unsigned long long>(p);
        asm volatile("prefetch.global.L2 [%0];" :: "l"(a));
    }
#else
    (void)p;(void)valid;
#endif
}
'''
s=s.replace(marker,helper+marker,1)
anchor='        const Count self7=v7?in[i7]:Count(0);\n'
if s.count(anchor)!=1: raise SystemExit(f'self7 anchor expected one match got {s.count(anchor)}')
lines=[anchor.rstrip('\n'),'        const Code next_base=base+Code(8)*grid;']
for k in range(8):
    # next loop base + k*grid. Avoid pointer arithmetic for invalid tails.
    lines.append(f'        const Code ni{k}=next_base+Code({k})*grid;')
    lines.append(f'        b300_prefetch_next_self_l2(ni{k}<n?in+ni{k}:in,ni{k}<n);')
insert='\n'.join(lines)+'\n'
s=s.replace(anchor,insert,1)
# The prefetch must remain before whichever cp.async wait the source uses.
pref=s.find('b300_prefetch_next_self_l2(ni7<n?in+ni7:in,ni7<n);')
waits=[x for x in (s.find('b300_cpasync_wait_pair();',pref),s.find('b300_cpasync_wait_oldest();',pref),s.find('b300_cpasync_wait_all();',pref)) if x>=0]
if 'cp.async.ca.shared.global' in s and (not waits or min(waits)<=pref):
    raise SystemExit('next-self prefetch is not before cp.async wait')
for req in ('prefetch.global.L2','const Code next_base=base+Code(8)*grid','b300_prefetch_next_self_l2(ni7<n?in+ni7:in,ni7<n)'):
    if req not in s: raise SystemExit(f'missing next-self prefetch artifact: {req}')
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(s)
print(f'generated {out} from {src}: b300_main_rankstate_ilp8_next_self_prefetch=1 next_iteration_self_prefetches_per_thread=8 cache=L2 prefetch_before_cpasync_wait=1 coalesced_per_k=1 semantics_unchanged=1 extra_shared_bytes=0')
