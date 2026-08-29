#!/usr/bin/env python3
from __future__ import annotations
import pathlib,sys

if len(sys.argv)!=4:
    raise SystemExit('usage: gen-b300-main-rankstate-hybrid-ilp8.py INPUT.cu OUTPUT.cu ILP8_MIN_STATES')
src=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2])
try: threshold=int(sys.argv[3],0)
except ValueError: raise SystemExit('ILP8_MIN_STATES must be a non-negative integer')
if threshold<0: raise SystemExit('ILP8_MIN_STATES must be non-negative')
s=src.read_text()
for req in (
    'b300_main_pull_rankstate_ilp4_kernel','b300_rankstate_ilp4_blocks',
    'b300_unpack_rank_delta','b300_unpack_rank_height','b300_pack_rank_state',
    'main_pull_direct_pair_source_rank','b300_add_rank_delta','b300_rank_delta_step',
    'b300_rank_height_advance'
):
    if req not in s: raise SystemExit(f'hybrid ILP8 requires artifact: {req}')
if 'b300_main_pull_rankstate_ilp8_kernel' in s:
    raise SystemExit('source already contains hybrid ILP8 kernel')

use_cg='b300_rankstate_random_load_cg' in s

def load(expr:str)->str:
    return f'b300_rankstate_random_load_cg({expr})' if use_cg else f'*({expr})'

def function_end(text:str,name:str)->int:
    p=text.find(name+'(')
    if p<0: raise SystemExit(f'function not found: {name}')
    brace=text.find('{',p);depth=0
    for i in range(brace,len(text)):
        if text[i]=='{': depth+=1
        elif text[i]=='}':
            depth-=1
            if depth==0: return i+1
    raise SystemExit(f'function end not found: {name}')

A=[];a=A.append
a('')
a('static inline int b300_main_rankstate_ilp8_blocks(Code n,int threads){')
a('    if(!n)return 1;')
a('    const Code cover=Code(threads)*8;')
a('    const Code need=(n+cover-1)/cover;')
a('    return int(std::min<Code>(65535,std::max<Code>(1,need)));')
a('}')
a('')
a('__global__ void b300_main_pull_rankstate_ilp8_kernel(')
a('    const Count* __restrict__ in,const MateID* __restrict__ mates,Code n,')
a('    const Count* __restrict__ in_block,Code nblock,')
a('    Count* __restrict__ out_main,int p,RankState* __restrict__ rank_state')
a('){')
a('    const Code tid=Code(blockIdx.x)*blockDim.x+threadIdx.x;')
a('    const Code grid=Code(gridDim.x)*blockDim.x;')
a('    for(Code base=tid;base<n;base+=8*grid){')
for k in range(8): a(f'        const Code i{k}=base+Code({k})*grid;')
for k in range(8): a('        const bool v0=true;' if k==0 else f'        const bool v{k}=i{k}<n;')
for k in range(8): a(f'        const MateID m{k}='+('mates[i0];' if k==0 else f'v{k}?mates[i{k}]:MateID(0);'))
for k in range(8): a(f'        const RankState s{k}='+('rank_state[i0];' if k==0 else f'v{k}?rank_state[i{k}]:RankState(0);'))
for k in range(8): a(f'        const RankDelta rd{k}='+('b300_unpack_rank_delta(s0);' if k==0 else f'v{k}?b300_unpack_rank_delta(s{k}):RankDelta(0);'))
for k in range(8): a(f'        const int h{k}='+('b300_unpack_rank_height(s0);' if k==0 else f'v{k}?b300_unpack_rank_height(s{k}):0;'))
for k in range(8): a(f'        const MateValuePair p{k}='+('mpair(m0,p);' if k==0 else f'v{k}?mpair(m{k},p):NN;'))
for k in range(8): a(f'        const bool hp{k}='+((f'(p0==LR||p0==NR||p0==NL);') if k==0 else f'v{k}&&(p{k}==LR||p{k}==NR||p{k}==NL);'))
for k in range(8): a(f'        const Code pj{k}=hp{k}?main_pull_direct_pair_source_rank(i{k},m{k},p,h{k}):Code(0);')
for k in range(8): a(f'        const MateValue mv{k}='+('mget(m0,p);' if k==0 else f'v{k}?mget(m{k},p):N;'))
for k in range(8): a(f'        const bool hb{k}='+((f'nblock&&mv0==N;') if k==0 else f'v{k}&&nblock&&mv{k}==N;'))
for k in range(8): a(f'        const Code bj{k}=hb{k}?b300_add_rank_delta(i{k},rd{k}):Code(0);')
a('')
for k in range(8):
    stmt=f'rank_state[i{k}]=b300_pack_rank_state(rd{k}+b300_rank_delta_step(mv{k},p,h{k}),b300_rank_height_advance(h{k},mv{k}));'
    a('        '+stmt if k==0 else f'        if(v{k})'+stmt)
a('')
for k in range(8): a(f'        const Count pair{k}=hp{k}?{load(f"in+pj{k}")}:Count(0);')
for k in range(8): a(f'        const Count block{k}=(hb{k}&&bj{k}<nblock)?{load(f"in_block+bj{k}")}:Count(0);')
for k in range(8): a(f'        const Count self{k}='+('in[i0];' if k==0 else f'v{k}?in[i{k}]:Count(0);'))
a('        const uint64_t mod=D_MOD;')
for k in range(8):
    stmt=f'uint64_t z=uint64_t(self{k})+pair{k}+block{k};if(z>=mod)z-=mod;if(z>=mod)z-=mod;out_main[i{k}]=Count(z);'
    a('        {'+stmt+'}' if k==0 else f'        if(v{k})'+'{'+stmt+'}')
a('    }')
a('}')
insert='\n'.join(A)+'\n'
end=function_end(s,'b300_main_pull_rankstate_ilp4_kernel')
s=s[:end]+insert+s[end:]

old='if(useRankDelta)b300_main_pull_rankstate_ilp4_kernel<<<b300_rankstate_ilp4_blocks(ms.size,threads),threads,0,c.sMain>>>(cur,c.dMate,ms.size,dcur,ds.size,nxt,p,c.dMainRankDelta);'
new=f'''if(useRankDelta){{
    if(ms.size>=Code({threshold}))
        b300_main_pull_rankstate_ilp8_kernel<<<b300_main_rankstate_ilp8_blocks(ms.size,threads),threads,0,c.sMain>>>(cur,c.dMate,ms.size,dcur,ds.size,nxt,p,c.dMainRankDelta);
    else
        b300_main_pull_rankstate_ilp4_kernel<<<b300_rankstate_ilp4_blocks(ms.size,threads),threads,0,c.sMain>>>(cur,c.dMate,ms.size,dcur,ds.size,nxt,p,c.dMainRankDelta);
}}'''
if s.count(old)!=1: raise SystemExit(f'hybrid launch anchor expected one match got {s.count(old)}')
s=s.replace(old,new,1)

for req in (
    'b300_main_pull_rankstate_ilp8_kernel','b300_main_rankstate_ilp8_blocks',
    'base+=8*grid','const Code i7=','const Count pair7=','const Count block7=',
    f'if(ms.size>=Code({threshold}))','b300_rankstate_ilp4_blocks(ms.size,threads)'
):
    if req not in s: raise SystemExit(f'missing hybrid ILP8 artifact: {req}')
if use_cg:
    for req in ('b300_rankstate_random_load_cg(in+pj7)','b300_rankstate_random_load_cg(in_block+bj7)'):
        if req not in s: raise SystemExit(f'missing hybrid ILP8 CG artifact: {req}')
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(s)
print(f'generated {out} from {src}: b300_main_rankstate_hybrid_ilp8=1 ilp8_min_states={threshold} ilp4_destinations=4 ilp8_destinations=8 random_cg={int(use_cg)} per_launch_state_threshold=1 register_pressure_isolated_by_kernel=1 requires_exact_ab=1')
