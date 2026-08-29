#!/usr/bin/env python3
from __future__ import annotations
import pathlib,sys

if len(sys.argv)!=4:
    raise SystemExit('usage: gen-b300-main-recurrence-ilp.py INPUT.cu OUTPUT.cu LANES(4|8)')
src=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);lanes=int(sys.argv[3])
if lanes not in (4,8):raise SystemExit('LANES must be 4 or 8')
s=src.read_text()
for req in (
    'main_pull_kernel_ilp2','b300_main_pull_prepare','b300_low_window_cache_active',
    'b300_high_main_state_active','b300_low_state_advance','b300_high_state_advance',
    'high_rec_groups='
):
    if req not in s:raise SystemExit(f'recurrence ILP{lanes} requires artifact: {req}')

def replace_function(text:str,name:str,new:str)->str:
    p=text.find(name+'(')
    if p<0:raise SystemExit(f'function not found: {name}')
    start=text.rfind('\n',0,p)+1;brace=text.find('{',p);depth=0;end=-1
    for i in range(brace,len(text)):
        if text[i]=='{':depth+=1
        elif text[i]=='}':
            depth-=1
            if depth==0:end=i+1;break
    if end<0:raise SystemExit(f'function end not found: {name}')
    return text[:start]+new+text[end:]

A=[];a=A.append
a('__global__ void main_pull_kernel_ilp2(')
a('    const Count* __restrict__ in,MateID* __restrict__ mates,Code n,')
a('    const Count* __restrict__ in_block,Code nblock,')
a('    Count* __restrict__ out_main,int p')
a('){')
a('    const Code tid=Code(blockIdx.x)*blockDim.x+threadIdx.x;')
a('    const Code grid=Code(gridDim.x)*blockDim.x;')
a('    const bool low=b300_low_window_cache_active(),high=b300_high_main_state_active();')
a(f'    for(Code base=tid;base<n;base+=Code({lanes})*grid)'+'{')
for k in range(lanes):
    a(f'        const Code i{k}=base+Code({k})*grid;')
for k in range(lanes):
    a('        const bool v0=true;' if k==0 else f'        const bool v{k}=i{k}<n;')
for k in range(lanes):
    a(f'        const MateID m{k}='+('mates[i0];' if k==0 else f'v{k}?mates[i{k}]:MateID(0);'))
for k in range(lanes):
    a(f'        Code pj{k}=0,bj{k}=0;bool hp{k}=false,hb{k}=false;')
for k in range(lanes):
    if k==0:a(f'        b300_main_pull_prepare(i0,m0,p,nblock,pj0,hp0,bj0,hb0);')
    else:a(f'        if(v{k})b300_main_pull_prepare(i{k},m{k},p,nblock,pj{k},hp{k},bj{k},hb{k});')
a('')
# Issue all irregular requests before the contiguous self stream.
for k in range(lanes):a(f'        const Count pair{k}=hp{k}?in[pj{k}]:Count(0);')
for k in range(lanes):a(f'        const Count block{k}=hb{k}?in_block[bj{k}]:Count(0);')
for k in range(lanes):
    a(f'        const Count self{k}='+('in[i0];' if k==0 else f'v{k}?in[i{k}]:Count(0);'))
a('        const uint64_t mod=D_MOD;')
for k in range(lanes):
    body=f'uint64_t z=uint64_t(self{k})+pair{k}+block{k};if(z>=mod)z-=mod;if(z>=mod)z-=mod;out_main[i{k}]=Count(z);if(low)mates[i{k}]=b300_low_state_advance(m{k},p);else if(high)mates[i{k}]=b300_high_state_advance(m{k},p);'
    if k==0:a('        {'+body+'}')
    else:a(f'        if(v{k})'+'{'+body+'}')
a('    }')
a('}')
new='\n'.join(A)
s=replace_function(s,'main_pull_kernel_ilp2',new)

for req in (
    f'base+=Code({lanes})*grid',f'const Code i{lanes-1}=',
    f'const Count pair{lanes-1}=',f'const Count block{lanes-1}=',
    f'const Count self{lanes-1}=',f'mates[i{lanes-1}]=b300_high_state_advance',
    'const bool low=b300_low_window_cache_active(),high=b300_high_main_state_active()'
):
    if req not in s:raise SystemExit(f'missing recurrence ILP{lanes} artifact: {req}')
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(s)
print(f'generated {out} from {src}: unified_main_recurrence_ilp={lanes} destinations_per_thread={lanes} extra_state_bytes=0 irregular_requests_first=1 recurrent_mate_updates={lanes} low_high_uniform_flags=1 register_pressure_high=1 requires_ab=1')
