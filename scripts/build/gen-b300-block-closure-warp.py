#!/usr/bin/env python3
from __future__ import annotations
import pathlib,sys

if len(sys.argv)!=3:
    raise SystemExit('usage: gen-b300-block-closure-warp.py INPUT.cu OUTPUT.cu')
src=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);s=src.read_text()
for req in ('b300_block_pull_rankstate_ilp4_kernel','b300_block_rankstate_ilp4_closure','block_pull_rank_contrib','block_pull_endpoint_mask','block_pull_apply_delta'):
    if req not in s:raise SystemExit(f'closure-warp transform requires generated artifact: {req}')

marker='\n\nstatic Code rank_full(MateID m,int width)'
if marker not in s:raise SystemExit('rank_full marker not found')
insert=r'''

// Each warp first classifies 32 blocked states in parallel. Only lanes whose
// symbol at p-1 is N enter the closure slow path. For each selected closure
// state lane 0 performs the exact incremental rank walk without source loads,
// stages up to 32 source ranks in shared memory, then the full warp issues the
// HBM reads concurrently and reduces them modulo D_MOD.
__global__ void b300_block_closure_warp_kernel(
    const Count* __restrict__ in_main,const MateID* __restrict__ block_mates,
    Code n,Count* __restrict__ out_block,int p,RankState* __restrict__ rank_state
){
    constexpr int MAX_WARPS=32,MAX_TERMS=32;
    __shared__ Code ranks[MAX_WARPS][MAX_TERMS];
    const unsigned lane=threadIdx.x&31u;
    const unsigned warp_in_block=threadIdx.x>>5;
    const unsigned mask=0xffffffffu;
    const Code block_thread_base=Code(blockIdx.x)*blockDim.x+(threadIdx.x&~31u);
    const Code grid_threads=Code(gridDim.x)*blockDim.x;
    for(Code base=block_thread_base;base<n;base+=grid_threads){
        const Code my_i=base+lane;
        const bool valid=my_i<n;
        const MateID my_b=valid?block_mates[my_i]:MateID(0);
        unsigned closure_mask=__ballot_sync(mask,valid&&mget(my_b,p-1)==N);
        while(closure_mask){
            const int owner=__ffs(closure_mask)-1;
            const Code i=base+Code(owner);
            const MateID b=__shfl_sync(mask,my_b,owner);
            RankState owner_state=0;
            if(int(lane)==owner)owner_state=rank_state[i];
            const RankState st=__shfl_sync(mask,owner_state,owner);
            int cnt=0;RankDelta next_rd=0;int next_h=0;
            if(lane==0){
                const RankDelta rd=b300_unpack_rank_delta(st);
                const int H=b300_unpack_rank_height(st);
                next_rd=rd+b300_rank_delta_step(N,p,H);
                next_h=b300_rank_height_advance(H,N);
                const MateID d=minsert(b,p-1,N);
                const Code base_rank=b300_sub_rank_delta(i,next_rd);
                auto emit=[&](Code r){if(cnt>=MAX_TERMS)asm("trap;");ranks[warp_in_block][cnt++]=r;};
                if(H>0){
                    const BlockClosureDelta x=BlockClosureDelta(block_pull_rank_contrib(R,p,H))+BlockClosureDelta(block_pull_rank_contrib(L,p-1,H-1));
                    emit(block_pull_apply_delta(base_rank,x));
                }
                const uint32_t endpoints=block_pull_endpoint_mask(d);
                BlockClosureDelta ldelta=BlockClosureDelta(block_pull_rank_contrib(L,p,H))+BlockClosureDelta(block_pull_rank_contrib(L,p-1,H+1));
                int hb=H,bal=0;uint32_t left=endpoints&((uint32_t(1)<<(p-1))-1u);
                while(left){
                    const int q=31-__clz(left);const MateValue v=mget(d,q);
                    if(bal==0&&v==L){const BlockClosureDelta x=ldelta+BlockClosureDelta(block_pull_rank_contrib(R,q,hb+2))-BlockClosureDelta(block_pull_rank_contrib(L,q,hb));emit(block_pull_apply_delta(base_rank,x));}
                    ldelta+=BlockClosureDelta(block_pull_rank_contrib(v,q,hb+2))-BlockClosureDelta(block_pull_rank_contrib(v,q,hb));hb=block_pull_advance_height(hb,v);if(v==L)++bal;else --bal;left^=uint32_t(1)<<q;if(bal<0)break;
                }
                BlockClosureDelta rsuffix=BlockClosureDelta(block_pull_rank_contrib(R,p,H+2))+BlockClosureDelta(block_pull_rank_contrib(R,p-1,H+1));
                int hbelow=H;bal=0;uint32_t right=endpoints&~((uint32_t(1)<<(p+1))-1u);
                while(right){
                    const int q=__ffs(right)-1;const MateValue v=mget(d,q);const int hq=hbelow+(v==R)-(v==L);
                    if(bal==0&&v==R){const BlockClosureDelta x=BlockClosureDelta(block_pull_rank_contrib(L,q,hq))-BlockClosureDelta(block_pull_rank_contrib(R,q,hq))+rsuffix;emit(block_pull_apply_delta(base_rank,x));}
                    rsuffix+=BlockClosureDelta(block_pull_rank_contrib(v,q,hq+2))-BlockClosureDelta(block_pull_rank_contrib(v,q,hq));hbelow=hq;if(v==R)++bal;else --bal;right&=right-1u;if(bal<0)break;
                }
            }
            cnt=__shfl_sync(mask,cnt,0);
            next_rd=__shfl_sync(mask,next_rd,0);
            next_h=__shfl_sync(mask,next_h,0);
            __syncwarp(mask);
            Count v=lane<unsigned(cnt)?in_main[ranks[warp_in_block][lane]]:Count(0);
            for(int off=16;off;off>>=1){
                const Count other=__shfl_down_sync(mask,v,off);
                if(lane<unsigned(off))pull_add_mod(v,other);
            }
            if(lane==0){out_block[i]=v;rank_state[i]=b300_pack_rank_state(next_rd,next_h);}
            __syncwarp(mask);
            closure_mask&=closure_mask-1u;
        }
    }
}
'''
s=s.replace(marker,insert+marker,1)

old=r'''        Count a0=endpoint0,a1=endpoint1,a2=endpoint2,a3=endpoint3;
        if(!ep0&&l0==N)a0=b300_block_rankstate_ilp4_closure(in_main,i0,b0,p,h0,nr0);
        if(v1&&!ep1&&l1==N)a1=b300_block_rankstate_ilp4_closure(in_main,i1,b1,p,h1,nr1);
        if(v2&&!ep2&&l2==N)a2=b300_block_rankstate_ilp4_closure(in_main,i2,b2,p,h2,nr2);
        if(v3&&!ep3&&l3==N)a3=b300_block_rankstate_ilp4_closure(in_main,i3,b3,p,h3,nr3);
        out_block[i0]=a0;rank_state[i0]=b300_pack_rank_state(nr0,b300_rank_height_advance(h0,l0));
        if(v1){out_block[i1]=a1;rank_state[i1]=b300_pack_rank_state(nr1,b300_rank_height_advance(h1,l1));}
        if(v2){out_block[i2]=a2;rank_state[i2]=b300_pack_rank_state(nr2,b300_rank_height_advance(h2,l2));}
        if(v3){out_block[i3]=a3;rank_state[i3]=b300_pack_rank_state(nr3,b300_rank_height_advance(h3,l3));}'''
new=r'''        // Endpoint states remain on the high-throughput per-thread ILP4 path.
        // N/closure states are deliberately untouched here and are completed by
        // the ballot-filtered warp closure kernel immediately afterwards.
        if(ep0){out_block[i0]=endpoint0;rank_state[i0]=b300_pack_rank_state(nr0,b300_rank_height_advance(h0,l0));}
        if(ep1){out_block[i1]=endpoint1;rank_state[i1]=b300_pack_rank_state(nr1,b300_rank_height_advance(h1,l1));}
        if(ep2){out_block[i2]=endpoint2;rank_state[i2]=b300_pack_rank_state(nr2,b300_rank_height_advance(h2,l2));}
        if(ep3){out_block[i3]=endpoint3;rank_state[i3]=b300_pack_rank_state(nr3,b300_rank_height_advance(h3,l3));}'''
if s.count(old)!=1:raise SystemExit(f'ILP4 closure body anchor expected one match got {s.count(old)}')
s=s.replace(old,new,1)

old='''if(ds.size){if(useRankDelta)b300_block_pull_rankstate_ilp4_kernel<<<bd,threads,0,c.sBlock>>>(cur,c.dBlockMate,ds.size,dnext,p,c.dBlockRankDelta);'''
new='''if(ds.size){if(useRankDelta){b300_block_pull_rankstate_ilp4_kernel<<<bd,threads,0,c.sBlock>>>(cur,c.dBlockMate,ds.size,dnext,p,c.dBlockRankDelta);b300_block_closure_warp_kernel<<<bd,threads,0,c.sBlock>>>(cur,c.dBlockMate,ds.size,dnext,p,c.dBlockRankDelta);}'''
if s.count(old)!=1:raise SystemExit(f'ILP4 launch anchor expected one match got {s.count(old)}')
s=s.replace(old,new,1)

for req in ('b300_block_closure_warp_kernel','__shared__ Code ranks[MAX_WARPS][MAX_TERMS]','__ballot_sync','closure_mask&=closure_mask-1u','lane<unsigned(cnt)?in_main','__shfl_down_sync','Endpoint states remain','b300_block_pull_rankstate_ilp4_kernel<<<','b300_block_closure_warp_kernel<<<','}else if(useBlockMate)'):
    if req not in s:raise SystemExit(f'missing closure-warp artifact: {req}')
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(s)
print(f'generated {out} from {src}: b300_block_closure_warp=1 warp_classification_batch=32 closure_ballot=1 rank_generation_lane0=1 source_loads_parallel_lanes=32 shared_bytes_per_block=8192 endpoint_kernel_split=1 closure_kernel_split=1 duplicate_mate_read_bytes_per_state=8 same_stream_exact_order=1')
