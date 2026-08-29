#!/usr/bin/env python3
from __future__ import annotations
import pathlib,sys

if len(sys.argv)!=3:
    raise SystemExit('usage: gen-b300-block-closure-warp-hybrid.py INPUT.cu OUTPUT.cu')
src=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);s=src.read_text()
for req in (
    'b300_block_closure_warp_kernel','b300_block_pull_rankstate_ilp4_kernel',
    'b300_block_rankstate_ilp4_closure','b300_rank_delta_step','b300_pack_rank_state',
    'block_pull_endpoint_mask','__ballot_sync','__shfl_down_sync'
):
    if req not in s:raise SystemExit(f'closure-warp hybrid requires artifact: {req}')

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

# Compile-time threshold: 0 reproduces all-warp closure scheduling. Positive
# values return closure states with fewer endpoint symbols to the scalar helper.
marker='\n\n// Each warp first classifies 32 blocked states in parallel.'
if marker not in s:raise SystemExit('closure warp comment marker not found')
helper=r'''

#ifndef B300_BLOCK_CLOSURE_WARP_MIN_ENDPOINTS
#define B300_BLOCK_CLOSURE_WARP_MIN_ENDPOINTS 0
#endif
static_assert(B300_BLOCK_CLOSURE_WARP_MIN_ENDPOINTS>=0 && B300_BLOCK_CLOSURE_WARP_MIN_ENDPOINTS<=28,
              "B300_BLOCK_CLOSURE_WARP_MIN_ENDPOINTS must be 0..28");
__device__ __forceinline__ bool b300_block_closure_warp_use_warp(MateID b,int p){
    if constexpr(B300_BLOCK_CLOSURE_WARP_MIN_ENDPOINTS==0)return true;
    const MateID d=minsert(b,p-1,N);
    return __popc(block_pull_endpoint_mask(d))>=B300_BLOCK_CLOSURE_WARP_MIN_ENDPOINTS;
}
'''
s=s.replace(marker,helper+marker,1)

# Endpoint kernel now also owns small closure states. Large closure states are
# deliberately untouched here so the following warp kernel remains their sole
# count/rank-state writer.
kernel=r'''__global__ void b300_block_pull_rankstate_ilp4_kernel(
    const Count* __restrict__ in_main,const MateID* __restrict__ block_mates,
    Code n,Count* __restrict__ out_block,int p,RankState* __restrict__ rank_state
){
    const Code tid=Code(blockIdx.x)*blockDim.x+threadIdx.x;
    const Code grid=Code(gridDim.x)*blockDim.x;
    for(Code base=tid;base<n;base+=4*grid){
        const Code i0=base,i1=base+grid,i2=base+2*grid,i3=base+3*grid;
        const bool v1=i1<n,v2=i2<n,v3=i3<n;
        const MateID b0=block_mates[i0],b1=v1?block_mates[i1]:MateID(0),b2=v2?block_mates[i2]:MateID(0),b3=v3?block_mates[i3]:MateID(0);
        const MateValue l0=mget(b0,p-1),l1=v1?mget(b1,p-1):N,l2=v2?mget(b2,p-1):N,l3=v3?mget(b3,p-1):N;
        const bool ep0=(l0==R||l0==L),ep1=v1&&(l1==R||l1==L),ep2=v2&&(l2==R||l2==L),ep3=v3&&(l3==R||l3==L);
        const bool cl0=(l0==N),cl1=v1&&(l1==N),cl2=v2&&(l2==N),cl3=v3&&(l3==N);
        const bool w0=cl0&&b300_block_closure_warp_use_warp(b0,p),w1=cl1&&b300_block_closure_warp_use_warp(b1,p),w2=cl2&&b300_block_closure_warp_use_warp(b2,p),w3=cl3&&b300_block_closure_warp_use_warp(b3,p);
        const bool sc0=cl0&&!w0,sc1=cl1&&!w1,sc2=cl2&&!w2,sc3=cl3&&!w3;
        const bool own0=ep0||sc0,own1=ep1||sc1,own2=ep2||sc2,own3=ep3||sc3;
        const RankState s0=own0?rank_state[i0]:RankState(0),s1=own1?rank_state[i1]:RankState(0),s2=own2?rank_state[i2]:RankState(0),s3=own3?rank_state[i3]:RankState(0);
        const RankDelta rd0=own0?b300_unpack_rank_delta(s0):RankDelta(0),rd1=own1?b300_unpack_rank_delta(s1):RankDelta(0),rd2=own2?b300_unpack_rank_delta(s2):RankDelta(0),rd3=own3?b300_unpack_rank_delta(s3):RankDelta(0);
        const int h0=own0?b300_unpack_rank_height(s0):0,h1=own1?b300_unpack_rank_height(s1):0,h2=own2?b300_unpack_rank_height(s2):0,h3=own3?b300_unpack_rank_height(s3):0;
        const RankDelta nr0=own0?rd0+b300_rank_delta_step(l0,p,h0):RankDelta(0),nr1=own1?rd1+b300_rank_delta_step(l1,p,h1):RankDelta(0),nr2=own2?rd2+b300_rank_delta_step(l2,p,h2):RankDelta(0),nr3=own3?rd3+b300_rank_delta_step(l3,p,h3):RankDelta(0);
        const Code j0=ep0?b300_sub_rank_delta(i0,rd0):Code(0),j1=ep1?b300_sub_rank_delta(i1,rd1):Code(0),j2=ep2?b300_sub_rank_delta(i2,rd2):Code(0),j3=ep3?b300_sub_rank_delta(i3,rd3):Code(0);
        const Count x0=ep0?in_main[j0]:Count(0),x1=ep1?in_main[j1]:Count(0),x2=ep2?in_main[j2]:Count(0),x3=ep3?in_main[j3]:Count(0);
        Count a0=x0,a1=x1,a2=x2,a3=x3;
        if(sc0)a0=b300_block_rankstate_ilp4_closure(in_main,i0,b0,p,h0,nr0);
        if(sc1)a1=b300_block_rankstate_ilp4_closure(in_main,i1,b1,p,h1,nr1);
        if(sc2)a2=b300_block_rankstate_ilp4_closure(in_main,i2,b2,p,h2,nr2);
        if(sc3)a3=b300_block_rankstate_ilp4_closure(in_main,i3,b3,p,h3,nr3);
        if(own0){out_block[i0]=a0;rank_state[i0]=b300_pack_rank_state(nr0,b300_rank_height_advance(h0,l0));}
        if(own1){out_block[i1]=a1;rank_state[i1]=b300_pack_rank_state(nr1,b300_rank_height_advance(h1,l1));}
        if(own2){out_block[i2]=a2;rank_state[i2]=b300_pack_rank_state(nr2,b300_rank_height_advance(h2,l2));}
        if(own3){out_block[i3]=a3;rank_state[i3]=b300_pack_rank_state(nr3,b300_rank_height_advance(h3,l3));}
    }
}'''
s=replace_function(s,'b300_block_pull_rankstate_ilp4_kernel',kernel)

# The warp kernel must use the identical predicate, making scalar and warp
# closure ownership a disjoint exhaustive partition.
old='unsigned closure_mask=__ballot_sync(mask,valid&&mget(my_b,p-1)==N);'
new='unsigned closure_mask=__ballot_sync(mask,valid&&mget(my_b,p-1)==N&&b300_block_closure_warp_use_warp(my_b,p));'
if s.count(old)!=1:raise SystemExit(f'closure ballot anchor expected once got {s.count(old)}')
s=s.replace(old,new,1)

for req in (
    'B300_BLOCK_CLOSURE_WARP_MIN_ENDPOINTS','b300_block_closure_warp_use_warp',
    'const bool sc0=cl0&&!w0','if(sc0)a0=b300_block_rankstate_ilp4_closure',
    'valid&&mget(my_b,p-1)==N&&b300_block_closure_warp_use_warp(my_b,p)',
    '__shfl_down_sync','b300_block_closure_warp_kernel<<<'
):
    if req not in s:raise SystemExit(f'closure-warp hybrid artifact missing: {req}')
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(s)
print(f'generated {out} from {src}: b300_block_closure_warp_hybrid=1 threshold_macro=B300_BLOCK_CLOSURE_WARP_MIN_ENDPOINTS scalar_small_closure=1 warp_large_closure=1 disjoint_owner_predicate=1 extra_state_bytes=0')
