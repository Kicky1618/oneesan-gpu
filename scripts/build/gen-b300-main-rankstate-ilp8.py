#!/usr/bin/env python3
from __future__ import annotations
import pathlib,sys

if len(sys.argv)!=3:
    raise SystemExit('usage: gen-b300-main-rankstate-ilp8.py INPUT.cu OUTPUT.cu')
src=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);s=src.read_text()
for req in (
    'b300_main_pull_rankstate_ilp4_kernel','b300_rankstate_ilp4_blocks',
    'b300_unpack_rank_delta','b300_unpack_rank_height','b300_pack_rank_state',
    'main_pull_direct_pair_source_rank','b300_add_rank_delta','b300_rank_delta_step',
    'b300_rank_height_advance'
):
    if req not in s:raise SystemExit(f'main ILP8 requires artifact: {req}')

use_cg='b300_rankstate_random_load_cg' in s

def load(expr:str)->str:
    return f'b300_rankstate_random_load_cg({expr})' if use_cg else f'*({expr})'

def replace_function(text:str,name:str,new:str)->str:
    p=text.find(name+'(')
    if p<0:raise SystemExit(f'function not found: {name}')
    start=text.rfind('\n',0,p)+1
    brace=text.find('{',p);depth=0;end=-1
    for k in range(brace,len(text)):
        if text[k]=='{':depth+=1
        elif text[k]=='}':
            depth-=1
            if depth==0:end=k+1;break
    if end<0:raise SystemExit(f'function end not found: {name}')
    return text[:start]+new+text[end:]

lines=[]
A=lines.append
A('__global__ void b300_main_pull_rankstate_ilp4_kernel(')
A('    const Count* __restrict__ in,const MateID* __restrict__ mates,Code n,')
A('    const Count* __restrict__ in_block,Code nblock,')
A('    Count* __restrict__ out_main,int p,RankState* __restrict__ rank_state')
A('){')
A('    const Code tid=Code(blockIdx.x)*blockDim.x+threadIdx.x;')
A('    const Code grid=Code(gridDim.x)*blockDim.x;')
A('    for(Code base=tid;base<n;base+=8*grid){')
for k in range(8):
    A(f'        const Code i{k}=base+Code({k})*grid;')
for k in range(8):
    if k==0:A('        const bool v0=true;')
    else:A(f'        const bool v{k}=i{k}<n;')
for k in range(8):
    if k==0:A('        const MateID m0=mates[i0];')
    else:A(f'        const MateID m{k}=v{k}?mates[i{k}]:MateID(0);')
for k in range(8):
    if k==0:A('        const RankState s0=rank_state[i0];')
    else:A(f'        const RankState s{k}=v{k}?rank_state[i{k}]:RankState(0);')
for k in range(8):
    A(f'        const RankDelta rd{k}=v{k}?b300_unpack_rank_delta(s{k}):RankDelta(0);')
for k in range(8):
    A(f'        const int h{k}=v{k}?b300_unpack_rank_height(s{k}):0;')
for k in range(8):
    A(f'        const MateValuePair p{k}=v{k}?mpair(m{k},p):NN;')
for k in range(8):
    A(f'        const bool hp{k}=v{k}&&(p{k}==LR||p{k}==NR||p{k}==NL);')
for k in range(8):
    A(f'        const Code pj{k}=hp{k}?main_pull_direct_pair_source_rank(i{k},m{k},p,h{k}):Code(0);')
for k in range(8):
    A(f'        const MateValue mv{k}=v{k}?mget(m{k},p):N;')
for k in range(8):
    A(f'        const bool hb{k}=v{k}&&nblock&&mv{k}==N;')
for k in range(8):
    A(f'        const Code bj{k}=hb{k}?b300_add_rank_delta(i{k},rd{k}):Code(0);')
A('')
for k in range(8):
    if k==0:
        A(f'        rank_state[i0]=b300_pack_rank_state(rd0+b300_rank_delta_step(mv0,p,h0),b300_rank_height_advance(h0,mv0));')
    else:
        A(f'        if(v{k})rank_state[i{k}]=b300_pack_rank_state(rd{k}+b300_rank_delta_step(mv{k},p,h{k}),b300_rank_height_advance(h{k},mv{k}));')
A('')
for k in range(8):
    A(f'        const Count pair{k}=hp{k}?{load(f"in+pj{k}")}:Count(0);')
for k in range(8):
    A(f'        const Count block{k}=(hb{k}&&bj{k}<nblock)?{load(f"in_block+bj{k}")}:Count(0);')
for k in range(8):
    if k==0:A('        const Count self0=in[i0];')
    else:A(f'        const Count self{k}=v{k}?in[i{k}]:Count(0);')
A('        const uint64_t mod=D_MOD;')
for k in range(8):
    stmt=f'uint64_t a=uint64_t(self{k})+pair{k}+block{k};if(a>=mod)a-=mod;if(a>=mod)a-=mod;out_main[i{k}]=Count(a);'
    if k==0:A('        {'+stmt+'}')
    else:A(f'        if(v{k})'+'{'+stmt+'}')
A('    }')
A('}')
new='\n'.join(lines)
s=replace_function(s,'b300_main_pull_rankstate_ilp4_kernel',new)

marker='static inline int b300_rankstate_ilp4_blocks(Code n,int threads){'
p=s.find(marker)
if p<0:raise SystemExit('ILP4 block helper not found')
insert='''static inline int b300_main_rankstate_ilp8_blocks(Code n,int threads){
    if(!n)return 1;
    const Code cover=Code(threads)*8;
    const Code need=(n+cover-1)/cover;
    return int(std::min<Code>(65535,std::max<Code>(1,need)));
}

'''
s=s[:p]+insert+s[p:]
old='b300_main_pull_rankstate_ilp4_kernel<<<b300_rankstate_ilp4_blocks(ms.size,threads),threads,0,c.sMain>>>'
new_launch='b300_main_pull_rankstate_ilp4_kernel<<<b300_main_rankstate_ilp8_blocks(ms.size,threads),threads,0,c.sMain>>>'
if s.count(old)!=1:raise SystemExit(f'main ILP4 launch expected one match got {s.count(old)}')
s=s.replace(old,new_launch,1)

for req in (
    'b300_main_rankstate_ilp8_blocks','Code(threads)*8','base+=8*grid',
    'const Code i7=','const Code pj7=','const Code bj7=',
    'const Count pair7=','const Count block7=','out_main[i7]=Count(a)',
    'b300_main_rankstate_ilp8_blocks(ms.size,threads)'
):
    if req not in s:raise SystemExit(f'missing ILP8 artifact: {req}')
if use_cg:
    for req in ('b300_rankstate_random_load_cg(in+pj7)','b300_rankstate_random_load_cg(in_block+bj7)'):
        if req not in s:raise SystemExit(f'missing ILP8 CG artifact: {req}')

out.parent.mkdir(parents=True,exist_ok=True);out.write_text(s)
print(f'generated {out} from {src}: b300_main_rankstate_ilp8=1 destinations_per_thread=8 main_random_count_requests_up_to=16 rankstate_updates=8 index_first=1 random_cg={int(use_cg)} block_path_unchanged=1 closure_path_unchanged=1 register_pressure_high=1 requires_register_ab=1')
