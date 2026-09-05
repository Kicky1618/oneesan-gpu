#!/usr/bin/env python3
from __future__ import annotations
import pathlib,sys

if len(sys.argv)!=3:
    raise SystemExit('usage: gen-b300-main-rankstate-ilp8-prefetch.py INPUT.cu OUTPUT.cu')
src=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);s=src.read_text()
for req in ('base+=8*grid','const Code pj7=','const Code bj7=','rank_state[i7]=b300_pack_rank_state','const Count pair7='):
    if req not in s:raise SystemExit(f'ILP8 prefetch requires artifact: {req}')
if 'cp.async.ca.shared.global' in s:
    raise SystemExit('ILP8 L2 prefetch and cp.async are separate experiments')

marker='\n\nstatic Code rank_full(MateID m,int width)'
if marker not in s:raise SystemExit('rank_full marker not found')
helper=r'''

__device__ __forceinline__ void b300_prefetch_count_l2(const Count* p,bool valid){
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
anchor='        const Code bj7=hb7?b300_add_rank_delta(i7,rd7):Code(0);\n'
if s.count(anchor)!=1:raise SystemExit(f'ILP8 bj7 anchor expected one match got {s.count(anchor)}')
lines=[]
for k in range(8):lines.append(f'        b300_prefetch_count_l2(in+pj{k},hp{k});')
for k in range(8):lines.append(f'        b300_prefetch_count_l2((hb{k}&&bj{k}<nblock)?in_block+bj{k}:in,hb{k}&&bj{k}<nblock);')
insert=anchor+'\n        // Address generation is complete. Start the HBM transactions before\n        // rank-state recurrence and consume them only afterwards.\n'+'\n'.join(lines)+'\n'
s=s.replace(anchor,insert,1)

for req in ('prefetch.global.L2','b300_prefetch_count_l2(in+pj7','b300_prefetch_count_l2((hb7&&bj7<nblock)?in_block+bj7:in'):
    if req not in s:raise SystemExit(f'missing ILP8 prefetch artifact: {req}')
# Keep a meaningful independent-work gap between prefetch and synchronous load.
pref=s.find('b300_prefetch_count_l2(in+pj7')
state=s.find('rank_state[i0]=b300_pack_rank_state',pref)
load=s.find('const Count pair0=',state)
if not(pref>=0 and state>pref and load>state):raise SystemExit('prefetch/state/load ordering not preserved')

out.parent.mkdir(parents=True,exist_ok=True);out.write_text(s)
print(f'generated {out} from {src}: b300_main_rankstate_ilp8_prefetch=1 prefetch_cache=L2 addresses_per_thread_max=16 prefetch_before_rankstate=1 synchronous_consume_after_rankstate=1 shared_bytes=0 semantics_unchanged=1')
