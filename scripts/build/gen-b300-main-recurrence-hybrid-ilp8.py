#!/usr/bin/env python3
from __future__ import annotations
import pathlib,sys

if len(sys.argv)!=4:
    raise SystemExit('usage: gen-b300-main-recurrence-hybrid-ilp8.py INPUT.cu OUTPUT.cu ILP8_MIN_STATES')
src=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2])
try: threshold=int(sys.argv[3],0)
except ValueError: raise SystemExit('ILP8_MIN_STATES must be a non-negative integer')
if threshold<0: raise SystemExit('ILP8_MIN_STATES must be non-negative')
s=src.read_text()
for req in (
    'main_pull_kernel_ilp2','b300_main_pull_ilp2_blocks','b300_main_pull_prepare',
    'b300_low_window_cache_active','b300_high_main_state_active',
    'b300_low_state_advance','b300_high_state_advance','high_rec_groups=',
    'unified_main_recurrence=1'
):
    # The last marker normally appears in the transform log rather than source;
    # tolerate its absence and rely on the recurrence helper artifacts above.
    if req=='unified_main_recurrence=1': continue
    if req not in s: raise SystemExit(f'hybrid recurrence ILP8 requires artifact: {req}')
if 'main_pull_kernel_ilp8_hybrid' in s:
    raise SystemExit('source already contains hybrid recurrence ILP8 kernel')

marker='\n\nstatic Code rank_full(MateID m,int width)'
if marker not in s: raise SystemExit('rank_full marker not found')

A=[];a=A.append
a('static inline int b300_main_recurrence_ilp8_hybrid_blocks(Code n,int threads){')
a('    if(!n)return 1;')
a('    const Code cover=Code(threads)*Code(8);')
a('    const Code need=(n+cover-1)/cover;')
a('    return int(std::min<Code>(65535,std::max<Code>(1,need)));')
a('}')
a('')
a('__global__ void main_pull_kernel_ilp8_hybrid(')
a('    const Count* __restrict__ in,MateID* __restrict__ mates,Code n,')
a('    const Count* __restrict__ in_block,Code nblock,')
a('    Count* __restrict__ out_main,int p')
a('){')
a('    const Code tid=Code(blockIdx.x)*blockDim.x+threadIdx.x;')
a('    const Code grid=Code(gridDim.x)*blockDim.x;')
a('    const bool low=b300_low_window_cache_active(),high=b300_high_main_state_active();')
a('    for(Code base=tid;base<n;base+=Code(8)*grid){')
for k in range(8): a(f'        const Code i{k}=base+Code({k})*grid;')
for k in range(8): a('        const bool v0=true;' if k==0 else f'        const bool v{k}=i{k}<n;')
for k in range(8): a(f'        const MateID m{k}='+('mates[i0];' if k==0 else f'v{k}?mates[i{k}]:MateID(0);'))
for k in range(8): a(f'        Code pj{k}=0,bj{k}=0;bool hp{k}=false,hb{k}=false;')
for k in range(8):
    if k==0: a('        b300_main_pull_prepare(i0,m0,p,nblock,pj0,hp0,bj0,hb0);')
    else: a(f'        if(v{k})b300_main_pull_prepare(i{k},m{k},p,nblock,pj{k},hp{k},bj{k},hb{k});')
a('')
# Issue irregular requests first.  Keeping them separate from self loads gives
# the Blackwell memory system up to 16 independent random Count requests/thread.
for k in range(8): a(f'        const Count pair{k}=hp{k}?in[pj{k}]:Count(0);')
for k in range(8): a(f'        const Count block{k}=hb{k}?in_block[bj{k}]:Count(0);')
for k in range(8): a(f'        const Count self{k}='+('in[i0];' if k==0 else f'v{k}?in[i{k}]:Count(0);'))
a('        const uint64_t mod=D_MOD;')
for k in range(8):
    body=(f'uint64_t z=uint64_t(self{k})+pair{k}+block{k};'
          f'if(z>=mod)z-=mod;if(z>=mod)z-=mod;out_main[i{k}]=Count(z);'
          f'if(low)mates[i{k}]=b300_low_state_advance(m{k},p);'
          f'else if(high)mates[i{k}]=b300_high_state_advance(m{k},p);')
    a('        {'+body+'}' if k==0 else f'        if(v{k})'+'{'+body+'}')
a('    }')
a('}')
insert='\n'.join(A)+'\n'
s=s.replace(marker,'\n\n'+insert+marker,1)

old='main_pull_kernel_ilp2<<<b300_main_pull_ilp2_blocks(ms.size,threads),threads,0,c.sMain>>>(cur,c.dMate,ms.size,dcur,ds.size,nxt,p)'
if s.count(old)!=1:
    raise SystemExit(f'hybrid recurrence launch anchor expected one scaled ILP2 match got {s.count(old)}')
new=f'''(ms.size>=Code({threshold})
                    ? (main_pull_kernel_ilp8_hybrid<<<b300_main_recurrence_ilp8_hybrid_blocks(ms.size,threads),threads,0,c.sMain>>>(cur,c.dMate,ms.size,dcur,ds.size,nxt,p), void())
                    : (main_pull_kernel_ilp2<<<b300_main_pull_ilp2_blocks(ms.size,threads),threads,0,c.sMain>>>(cur,c.dMate,ms.size,dcur,ds.size,nxt,p), void()))'''
# CUDA kernel launches are statements and cannot be used as ordinary C++
# conditional-expression operands. Replace the enclosing one-line if instead.
old_stmt='if(useMate)'+old+';'
new_stmt=f'''if(useMate){{
                if(ms.size>=Code({threshold})) main_pull_kernel_ilp8_hybrid<<<b300_main_recurrence_ilp8_hybrid_blocks(ms.size,threads),threads,0,c.sMain>>>(cur,c.dMate,ms.size,dcur,ds.size,nxt,p);
                else main_pull_kernel_ilp2<<<b300_main_pull_ilp2_blocks(ms.size,threads),threads,0,c.sMain>>>(cur,c.dMate,ms.size,dcur,ds.size,nxt,p);
            }}'''
if s.count(old_stmt)!=1:
    raise SystemExit(f'hybrid recurrence enclosing launch expected one match got {s.count(old_stmt)}')
s=s.replace(old_stmt,new_stmt,1)

for req in (
    'main_pull_kernel_ilp2','main_pull_kernel_ilp8_hybrid',
    'b300_main_pull_ilp2_blocks(ms.size,threads)',
    'b300_main_recurrence_ilp8_hybrid_blocks(ms.size,threads)',
    'base+=Code(8)*grid','const Code i7=','const Count pair7=',
    'const Count block7=','const Count self7=',
    'mates[i7]=b300_high_state_advance',f'if(ms.size>=Code({threshold}))',
):
    if req not in s: raise SystemExit(f'missing hybrid recurrence ILP8 artifact: {req}')
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(s)
print(f'generated {out} from {src}: b300_main_recurrence_hybrid_ilp8=1 base_ilp=2 high_ilp=8 ilp8_min_states={threshold} ilp2_launch=ceil_n_over_2threads_capped65535 ilp8_launch=ceil_n_over_8threads_capped65535 separate_kernels=1 register_pressure_isolated=1 batch_abi_preserved=1 requires_exact_ab=1')
