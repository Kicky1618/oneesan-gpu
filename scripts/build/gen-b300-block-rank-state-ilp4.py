#!/usr/bin/env python3
from __future__ import annotations
import pathlib,sys

if len(sys.argv)!=3:
    raise SystemExit('usage: gen-b300-block-rank-state-ilp4.py INPUT.cu OUTPUT.cu')
src=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);s=src.read_text()
for req in (
    'using RankState = unsigned long long;',
    'b300_unpack_rank_delta','b300_unpack_rank_height','b300_pack_rank_state',
    'block_pull_rank_contrib','block_pull_endpoint_mask','block_pull_apply_delta',
    'block_pull_kernel<true,true>'
):
    if req not in s: raise SystemExit(f'block rank-state ILP4 requires generated artifact: {req}')

marker='\n\nstatic Code rank_full(MateID m,int width)'
if marker not in s: raise SystemExit('rank_full marker not found')
insert=r'''

#ifndef B300_BLOCK_CLOSURE_QUAD
#define B300_BLOCK_CLOSURE_QUAD 0
#endif
#ifndef B300_BLOCK_CLOSURE_CG
#define B300_BLOCK_CLOSURE_CG 0
#endif
static_assert(B300_BLOCK_CLOSURE_QUAD==0||B300_BLOCK_CLOSURE_QUAD==1,"B300_BLOCK_CLOSURE_QUAD must be 0 or 1");
static_assert(B300_BLOCK_CLOSURE_CG==0||B300_BLOCK_CLOSURE_CG==1,"B300_BLOCK_CLOSURE_CG must be 0 or 1");
static_assert(!B300_BLOCK_CLOSURE_CG || B300_BLOCK_CLOSURE_QUAD,
              "B300_BLOCK_CLOSURE_CG requires B300_BLOCK_CLOSURE_QUAD");
static_assert(sizeof(Count)==4,"B300 block closure payload assumes 32-bit Count");

__device__ __forceinline__ Count b300_block_closure_payload_load(const Count* p){
#if B300_BLOCK_CLOSURE_CG && __CUDA_ARCH__
    uint32_t v;const unsigned long long a=reinterpret_cast<unsigned long long>(p);
    asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(a));
    return Count(v);
#else
    return __ldg(p);
#endif
}

struct B300ClosureQuadRanks{Code r0=0,r1=0,r2=0,r3=0;int n=0;};
__device__ __forceinline__ void b300_closure_quad_flush(
    Count& acc,const Count* __restrict__ in_main,B300ClosureQuadRanks& q
){
#if B300_BLOCK_CLOSURE_QUAD
    // Keep the four independent payload loads adjacent so the scheduler can
    // maintain multiple outstanding misses.  Optional .cg bypasses L1 for this
    // large/random stream while retaining L2 caching.
    const Count v0=q.n>0?b300_block_closure_payload_load(in_main+q.r0):Count(0);
    const Count v1=q.n>1?b300_block_closure_payload_load(in_main+q.r1):Count(0);
    const Count v2=q.n>2?b300_block_closure_payload_load(in_main+q.r2):Count(0);
    const Count v3=q.n>3?b300_block_closure_payload_load(in_main+q.r3):Count(0);
    if(q.n>0)pull_add_mod(acc,v0);if(q.n>1)pull_add_mod(acc,v1);if(q.n>2)pull_add_mod(acc,v2);if(q.n>3)pull_add_mod(acc,v3);
    q.n=0;
#else
    (void)acc;(void)in_main;(void)q;
#endif
}
__device__ __forceinline__ void b300_closure_quad_emit(
    Count& acc,const Count* __restrict__ in_main,B300ClosureQuadRanks& q,Code rank
){
#if B300_BLOCK_CLOSURE_QUAD
    if(q.n==0)q.r0=rank;else if(q.n==1)q.r1=rank;else if(q.n==2)q.r2=rank;else q.r3=rank;
    ++q.n;if(q.n==4)b300_closure_quad_flush(acc,in_main,q);
#else
    block_pull_add_rank(acc,in_main,rank);
#endif
}

__device__ __noinline__ Count b300_block_rankstate_ilp4_closure(
    const Count* __restrict__ in_main,Code i,MateID b,int p,int H,RankDelta next_rd
){
    Count acc=0;B300ClosureQuadRanks quad{};
    MateID d=minsert(b,p-1,N);
    const Code base_rank=b300_sub_rank_delta(i,next_rd);
    if(H>0){
        const BlockClosureDelta rd=BlockClosureDelta(block_pull_rank_contrib(R,p,H))+BlockClosureDelta(block_pull_rank_contrib(L,p-1,H-1));
        b300_closure_quad_emit(acc,in_main,quad,block_pull_apply_delta(base_rank,rd));
    }
    const uint32_t endpoints=block_pull_endpoint_mask(d);
    BlockClosureDelta ldelta=BlockClosureDelta(block_pull_rank_contrib(L,p,H))+BlockClosureDelta(block_pull_rank_contrib(L,p-1,H+1));
    int hb=H,bal=0;uint32_t left=endpoints&((uint32_t(1)<<(p-1))-1u);
    while(left){
        const int q=31-__clz(left);const MateValue v=mget(d,q);
        if(bal==0&&v==L){const BlockClosureDelta xd=ldelta+BlockClosureDelta(block_pull_rank_contrib(R,q,hb+2))-BlockClosureDelta(block_pull_rank_contrib(L,q,hb));b300_closure_quad_emit(acc,in_main,quad,block_pull_apply_delta(base_rank,xd));}
        ldelta+=BlockClosureDelta(block_pull_rank_contrib(v,q,hb+2))-BlockClosureDelta(block_pull_rank_contrib(v,q,hb));hb=block_pull_advance_height(hb,v);if(v==L)++bal;else --bal;left^=uint32_t(1)<<q;if(bal<0)break;
    }
    BlockClosureDelta rsuffix=BlockClosureDelta(block_pull_rank_contrib(R,p,H+2))+BlockClosureDelta(block_pull_rank_contrib(R,p-1,H+1));
    int hbelow=H;bal=0;uint32_t right=endpoints&~((uint32_t(1)<<(p+1))-1u);
    while(right){
        const int q=__ffs(right)-1;const MateValue v=mget(d,q);const int hq=hbelow+(v==R)-(v==L);
        if(bal==0&&v==R){const BlockClosureDelta xd=BlockClosureDelta(block_pull_rank_contrib(L,q,hq))-BlockClosureDelta(block_pull_rank_contrib(R,q,hq))+rsuffix;b300_closure_quad_emit(acc,in_main,quad,block_pull_apply_delta(base_rank,xd));}
        rsuffix+=BlockClosureDelta(block_pull_rank_contrib(v,q,hq+2))-BlockClosureDelta(block_pull_rank_contrib(v,q,hq));hbelow=hq;if(v==R)++bal;else --bal;right&=right-1u;if(bal<0)break;
    }
    b300_closure_quad_flush(acc,in_main,quad);
    return acc;
}

__global__ void b300_block_pull_rankstate_ilp4_kernel(
    const Count* __restrict__ in_main,const MateID* __restrict__ block_mates,
    Code n,Count* __restrict__ out_block,int p,RankState* __restrict__ rank_state
){
    const Code tid=Code(blockIdx.x)*blockDim.x+threadIdx.x;
    const Code grid=Code(gridDim.x)*blockDim.x;
    for(Code base=tid;base<n;base+=4*grid){
        const Code i0=base,i1=base+grid,i2=base+2*grid,i3=base+3*grid;
        const bool v1=i1<n,v2=i2<n,v3=i3<n;
        const MateID b0=block_mates[i0],b1=v1?block_mates[i1]:MateID(0),b2=v2?block_mates[i2]:MateID(0),b3=v3?block_mates[i3]:MateID(0);
        const RankState s0=rank_state[i0],s1=v1?rank_state[i1]:RankState(0),s2=v2?rank_state[i2]:RankState(0),s3=v3?rank_state[i3]:RankState(0);
        const RankDelta rd0=b300_unpack_rank_delta(s0),rd1=v1?b300_unpack_rank_delta(s1):RankDelta(0),rd2=v2?b300_unpack_rank_delta(s2):RankDelta(0),rd3=v3?b300_unpack_rank_delta(s3):RankDelta(0);
        const int h0=b300_unpack_rank_height(s0),h1=v1?b300_unpack_rank_height(s1):0,h2=v2?b300_unpack_rank_height(s2):0,h3=v3?b300_unpack_rank_height(s3):0;
        const MateValue l0=mget(b0,p-1),l1=v1?mget(b1,p-1):N,l2=v2?mget(b2,p-1):N,l3=v3?mget(b3,p-1):N;
        const bool ep0=(l0==R||l0==L),ep1=v1&&(l1==R||l1==L),ep2=v2&&(l2==R||l2==L),ep3=v3&&(l3==R||l3==L);
        const RankDelta nr0=rd0+b300_rank_delta_step(l0,p,h0),nr1=v1?rd1+b300_rank_delta_step(l1,p,h1):RankDelta(0),nr2=v2?rd2+b300_rank_delta_step(l2,p,h2):RankDelta(0),nr3=v3?rd3+b300_rank_delta_step(l3,p,h3):RankDelta(0);
        const Code j0=ep0?b300_sub_rank_delta(i0,rd0):Code(0),j1=ep1?b300_sub_rank_delta(i1,rd1):Code(0),j2=ep2?b300_sub_rank_delta(i2,rd2):Code(0),j3=ep3?b300_sub_rank_delta(i3,rd3):Code(0);

        rank_state[i0]=b300_pack_rank_state(nr0,b300_rank_height_advance(h0,l0));
        if(v1)rank_state[i1]=b300_pack_rank_state(nr1,b300_rank_height_advance(h1,l1));
        if(v2)rank_state[i2]=b300_pack_rank_state(nr2,b300_rank_height_advance(h2,l2));
        if(v3)rank_state[i3]=b300_pack_rank_state(nr3,b300_rank_height_advance(h3,l3));

        const Count endpoint0=ep0?in_main[j0]:Count(0),endpoint1=ep1?in_main[j1]:Count(0),endpoint2=ep2?in_main[j2]:Count(0),endpoint3=ep3?in_main[j3]:Count(0);
        Count a0=endpoint0,a1=endpoint1,a2=endpoint2,a3=endpoint3;
        if(!ep0&&l0==N)a0=b300_block_rankstate_ilp4_closure(in_main,i0,b0,p,h0,nr0);
        if(v1&&!ep1&&l1==N)a1=b300_block_rankstate_ilp4_closure(in_main,i1,b1,p,h1,nr1);
        if(v2&&!ep2&&l2==N)a2=b300_block_rankstate_ilp4_closure(in_main,i2,b2,p,h2,nr2);
        if(v3&&!ep3&&l3==N)a3=b300_block_rankstate_ilp4_closure(in_main,i3,b3,p,h3,nr3);
        out_block[i0]=a0;
        if(v1)out_block[i1]=a1;
        if(v2)out_block[i2]=a2;
        if(v3)out_block[i3]=a3;
    }
}
'''
s=s.replace(marker,insert+marker,1)
old='''if(ds.size){if(useRankDelta)block_pull_kernel<true,true><<<bd,threads,0,c.sBlock>>>(cur,c.dBlockMate,ds.size,dnext,p,c.dBlockRankDelta);'''
new='''if(ds.size){if(useRankDelta)b300_block_pull_rankstate_ilp4_kernel<<<bd,threads,0,c.sBlock>>>(cur,c.dBlockMate,ds.size,dnext,p,c.dBlockRankDelta);'''
if s.count(old)!=1: raise SystemExit(f'block rank-state ILP4 launch anchor expected one match got {s.count(old)}')
s=s.replace(old,new,1)
for req in (
    'b300_block_pull_rankstate_ilp4_kernel','base+=4*grid','endpoint3=',
    'b300_block_rankstate_ilp4_closure','B300_BLOCK_CLOSURE_QUAD','B300_BLOCK_CLOSURE_CG',
    'ld.global.cg.u32','b300_closure_quad_emit','rank_state[i3]=b300_pack_rank_state'
):
    if req not in s: raise SystemExit(f'missing block rank-state ILP4 artifact: {req}')
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(s)
print(f'generated {out} from {src}: b300_block_rank_state_ilp4=1 destinations_per_thread=4 endpoint_loads_issued_before_closure=1 closure_slow_path=noinline closure_register_isolation=1 closure_quad_compile_switch=B300_BLOCK_CLOSURE_QUAD closure_cg_compile_switch=B300_BLOCK_CLOSURE_CG packed_rank_state=1 rank_state_store_before_count_gather=1 register_live_range_reduced=1 register_pressure_requires_ab=1')
