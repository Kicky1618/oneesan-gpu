#!/usr/bin/env python3
from __future__ import annotations
import pathlib,sys

if len(sys.argv)!=3:
    raise SystemExit('usage: gen-b300-rankstate-ilp4-random-cg.py INPUT.cu OUTPUT.cu')
src=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);s=src.read_text()
for req in ('b300_main_pull_rankstate_ilp4_kernel','b300_block_pull_rankstate_ilp4_kernel','b300_block_closure_warp_kernel','using Count = uint32_t;'):
    if req not in s:raise SystemExit(f'ILP4 random-cg transform requires artifact: {req}')

marker='\n\nstatic Code rank_full(MateID m,int width)'
if marker not in s:raise SystemExit('rank_full marker missing')
helper=r'''

__device__ __forceinline__ Count b300_rankstate_random_load_cg(const Count* p){
#if defined(__CUDA_ARCH__)
    uint32_t v;
    const unsigned long long a=reinterpret_cast<unsigned long long>(p);
    asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(a));
    return Count(v);
#else
    return *p;
#endif
}
'''
s=s.replace(marker,helper+marker,1)

replacements=(
('''        const Count pair0=hp0?in[pj0]:Count(0),pair1=hp1?in[pj1]:Count(0),pair2=hp2?in[pj2]:Count(0),pair3=hp3?in[pj3]:Count(0);''',
 '''        const Count pair0=hp0?b300_rankstate_random_load_cg(in+pj0):Count(0),pair1=hp1?b300_rankstate_random_load_cg(in+pj1):Count(0),pair2=hp2?b300_rankstate_random_load_cg(in+pj2):Count(0),pair3=hp3?b300_rankstate_random_load_cg(in+pj3):Count(0);''',
 'main pair random loads'),
('''        const Count block0=(hb0&&bj0<nblock)?in_block[bj0]:Count(0),block1=(hb1&&bj1<nblock)?in_block[bj1]:Count(0),block2=(hb2&&bj2<nblock)?in_block[bj2]:Count(0),block3=(hb3&&bj3<nblock)?in_block[bj3]:Count(0);''',
 '''        const Count block0=(hb0&&bj0<nblock)?b300_rankstate_random_load_cg(in_block+bj0):Count(0),block1=(hb1&&bj1<nblock)?b300_rankstate_random_load_cg(in_block+bj1):Count(0),block2=(hb2&&bj2<nblock)?b300_rankstate_random_load_cg(in_block+bj2):Count(0),block3=(hb3&&bj3<nblock)?b300_rankstate_random_load_cg(in_block+bj3):Count(0);''',
 'main blocked-preimage random loads'),
('''        const Count x0=ep0?in_main[j0]:Count(0),x1=ep1?in_main[j1]:Count(0),x2=ep2?in_main[j2]:Count(0),x3=ep3?in_main[j3]:Count(0);''',
 '''        const Count x0=ep0?b300_rankstate_random_load_cg(in_main+j0):Count(0),x1=ep1?b300_rankstate_random_load_cg(in_main+j1):Count(0),x2=ep2?b300_rankstate_random_load_cg(in_main+j2):Count(0),x3=ep3?b300_rankstate_random_load_cg(in_main+j3):Count(0);''',
 'block endpoint random loads'),
('''            Count v=lane<unsigned(cnt)?in_main[ranks[warp_in_block][lane]]:Count(0);''',
 '''            Count v=lane<unsigned(cnt)?b300_rankstate_random_load_cg(in_main+ranks[warp_in_block][lane]):Count(0);''',
 'closure warp random loads'),
)
for old,new,label in replacements:
    n=s.count(old)
    if n!=1:raise SystemExit(f'{label}: expected one match got {n}')
    s=s.replace(old,new,1)

# The coalesced destination self loads are deliberately left as normal loads so
# they can use the ordinary cache path. Only irregular predecessor gathers use
# .cg to avoid L1 pollution.
for req in ('ld.global.cg.u32','b300_rankstate_random_load_cg(in+pj0)','b300_rankstate_random_load_cg(in_block+bj0)','b300_rankstate_random_load_cg(in_main+j0)','b300_rankstate_random_load_cg(in_main+ranks[warp_in_block][lane])','const Count self0=in[i0]'):
    if req not in s:raise SystemExit(f'missing random-cg artifact: {req}')
for stale in ('pair0=hp0?in[pj0]','block0=(hb0&&bj0<nblock)?in_block[bj0]','x0=ep0?in_main[j0]','lane<unsigned(cnt)?in_main[ranks[warp_in_block][lane]]'):
    if stale in s:raise SystemExit(f'stale random load remains: {stale}')

out.parent.mkdir(parents=True,exist_ok=True);out.write_text(s)
print(f'generated {out} from {src}: b300_rankstate_ilp4_random_cg=1 random_load_cache=L2_only self_load_cache=normal main_pair=4 main_block=4 block_endpoint=4 closure_warp=32 exact_load_semantics=1')
