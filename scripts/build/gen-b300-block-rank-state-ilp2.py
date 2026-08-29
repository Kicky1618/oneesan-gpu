#!/usr/bin/env python3
from __future__ import annotations
import pathlib,sys

if len(sys.argv)!=3:
    raise SystemExit('usage: gen-b300-block-rank-state-ilp2.py INPUT.cu OUTPUT.cu')
src=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);s=src.read_text()
for req in (
    'using RankState = unsigned long long;',
    'b300_unpack_rank_delta', 'b300_unpack_rank_height', 'b300_pack_rank_state',
    'block_pull_rank_contrib', 'block_pull_endpoint_mask', 'block_pull_apply_delta',
    'block_pull_kernel<true,true>'
):
    if req not in s: raise SystemExit(f'block rank-state ILP2 requires generated artifact: {req}')

marker='\n\nstatic Code rank_full(MateID m,int width)'
if marker not in s: raise SystemExit('rank_full marker not found')
insert=r'''

// Keep the variable-length closure walk out of the common endpoint kernel's
// register allocation. About two thirds of W=28 blocked destinations take the
// one-source endpoint path, so a device call on the minority closure path is a
// better trade than making every resident warp carry closure temporaries.
__device__ __noinline__ Count b300_block_rankstate_closure(
    const Count* __restrict__ in_main,Code i,MateID b,int p,int H,RankDelta next_rd
){
    Count acc=0;
    MateID d=minsert(b,p-1,N);
    const Code base_rank=b300_sub_rank_delta(i,next_rd);

    if(H>0){
        const BlockClosureDelta rd=
            BlockClosureDelta(block_pull_rank_contrib(R,p,H))+
            BlockClosureDelta(block_pull_rank_contrib(L,p-1,H-1));
        block_pull_add_rank(acc,in_main,block_pull_apply_delta(base_rank,rd));
    }

    const uint32_t endpoints=block_pull_endpoint_mask(d);
    BlockClosureDelta ldelta=
        BlockClosureDelta(block_pull_rank_contrib(L,p,H))+
        BlockClosureDelta(block_pull_rank_contrib(L,p-1,H+1));
    int hb=H,bal=0;
    uint32_t left=endpoints&((uint32_t(1)<<(p-1))-1u);
    while(left){
        const int q=31-__clz(left);const MateValue v=mget(d,q);
        if(bal==0&&v==L){
            const BlockClosureDelta xd=ldelta+
                BlockClosureDelta(block_pull_rank_contrib(R,q,hb+2))-
                BlockClosureDelta(block_pull_rank_contrib(L,q,hb));
            block_pull_add_rank(acc,in_main,block_pull_apply_delta(base_rank,xd));
        }
        ldelta+=BlockClosureDelta(block_pull_rank_contrib(v,q,hb+2))-
                BlockClosureDelta(block_pull_rank_contrib(v,q,hb));
        hb=block_pull_advance_height(hb,v);
        if(v==L)++bal;else --bal;
        left^=uint32_t(1)<<q;
        if(bal<0)break;
    }

    BlockClosureDelta rsuffix=
        BlockClosureDelta(block_pull_rank_contrib(R,p,H+2))+
        BlockClosureDelta(block_pull_rank_contrib(R,p-1,H+1));
    int hbelow=H;bal=0;
    uint32_t right=endpoints&~((uint32_t(1)<<(p+1))-1u);
    while(right){
        const int q=__ffs(right)-1;const MateValue v=mget(d,q);
        const int hq=hbelow+(v==R)-(v==L);
        if(bal==0&&v==R){
            const BlockClosureDelta xd=
                BlockClosureDelta(block_pull_rank_contrib(L,q,hq))-
                BlockClosureDelta(block_pull_rank_contrib(R,q,hq))+rsuffix;
            block_pull_add_rank(acc,in_main,block_pull_apply_delta(base_rank,xd));
        }
        rsuffix+=BlockClosureDelta(block_pull_rank_contrib(v,q,hq+2))-
                 BlockClosureDelta(block_pull_rank_contrib(v,q,hq));
        hbelow=hq;
        if(v==R)++bal;else --bal;
        right&=right-1u;
        if(bal<0)break;
    }
    return acc;
}

__global__ void b300_block_pull_rankstate_ilp2_kernel(
    const Count* __restrict__ in_main,const MateID* __restrict__ block_mates,
    Code n,Count* __restrict__ out_block,int p,RankState* __restrict__ rank_state
){
    const Code tid=Code(blockIdx.x)*blockDim.x+threadIdx.x;
    const Code grid=Code(gridDim.x)*blockDim.x;
    for(Code base=tid;base<n;base+=2*grid){
        const Code i0=base,i1=base+grid;const bool v1=i1<n;
        const MateID b0=block_mates[i0],b1=v1?block_mates[i1]:MateID(0);
        const RankState s0=rank_state[i0],s1=v1?rank_state[i1]:RankState(0);
        const RankDelta rd0=b300_unpack_rank_delta(s0),rd1=v1?b300_unpack_rank_delta(s1):RankDelta(0);
        const int h0=b300_unpack_rank_height(s0),h1=v1?b300_unpack_rank_height(s1):0;
        const MateValue l0=mget(b0,p-1),l1=v1?mget(b1,p-1):N;
        const bool ep0=(l0==R||l0==L),ep1=v1&&(l1==R||l1==L);
        const RankDelta nr0=rd0+b300_rank_delta_step(l0,p,h0);
        const RankDelta nr1=v1?rd1+b300_rank_delta_step(l1,p,h1):RankDelta(0);

        const Code j0=ep0?b300_sub_rank_delta(i0,rd0):Code(0);
        const Code j1=ep1?b300_sub_rank_delta(i1,rd1):Code(0);
        const Count endpoint0=ep0?in_main[j0]:Count(0);
        const Count endpoint1=ep1?in_main[j1]:Count(0);

        Count a0=endpoint0;
        if(!ep0&&l0==N)a0=b300_block_rankstate_closure(in_main,i0,b0,p,h0,nr0);
        Count a1=endpoint1;
        if(v1&&!ep1&&l1==N)a1=b300_block_rankstate_closure(in_main,i1,b1,p,h1,nr1);

        out_block[i0]=a0;
        rank_state[i0]=b300_pack_rank_state(nr0,b300_rank_height_advance(h0,l0));
        if(v1){
            out_block[i1]=a1;
            rank_state[i1]=b300_pack_rank_state(nr1,b300_rank_height_advance(h1,l1));
        }
    }
}
'''
s=s.replace(marker,insert+marker,1)

old='''if(ds.size){if(useRankDelta)block_pull_kernel<true,true><<<bd,threads,0,c.sBlock>>>(cur,c.dBlockMate,ds.size,dnext,p,c.dBlockRankDelta);'''
new='''if(ds.size){if(useRankDelta)b300_block_pull_rankstate_ilp2_kernel<<<bd,threads,0,c.sBlock>>>(cur,c.dBlockMate,ds.size,dnext,p,c.dBlockRankDelta);'''
if s.count(old)!=1: raise SystemExit(f'block rank-state ILP2 launch anchor expected one match got {s.count(old)}')
s=s.replace(old,new,1)

for req in ('__noinline__ Count b300_block_rankstate_closure','b300_block_pull_rankstate_ilp2_kernel','base+=2*grid','const Count endpoint0=','const Count endpoint1=','rank_state[i0]=b300_pack_rank_state','rank_state[i1]=b300_pack_rank_state'):
    if req not in s: raise SystemExit(f'missing block rank-state ILP2 artifact: {req}')
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(s)
print(f'generated {out} from {src}: b300_block_rank_state_ilp2=1 destinations_per_thread=2 endpoint_loads_issued_before_closure=1 closure_slow_path=noinline closure_register_isolation=1 packed_rank_state=1')
