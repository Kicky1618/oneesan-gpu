#!/usr/bin/env python3
from __future__ import annotations
import pathlib,sys

if len(sys.argv)!=3:
    raise SystemExit('usage: gen-b300-block-closure-warp-parallel-prefix.py INPUT.cu OUTPUT.cu')
src=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);s=src.read_text()
for req in (
    'b300_block_closure_warp_kernel','b300_block_pull_rankstate_ilp4_kernel',
    'block_pull_rank_contrib','block_pull_apply_delta','b300_sub_rank_delta',
    'b300_unpack_rank_delta','b300_unpack_rank_height','b300_pack_rank_state'
):
    if req not in s:raise SystemExit(f'parallel closure prefix requires artifact: {req}')

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

new_kernel=r'''__global__ void b300_block_closure_warp_kernel(
    const Count* __restrict__ in_main,const MateID* __restrict__ block_mates,
    Code n,Count* __restrict__ out_block,int p,RankState* __restrict__ rank_state
){
    const unsigned lane=threadIdx.x&31u;
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
            const RankDelta rd=b300_unpack_rank_delta(st);
            const int H=b300_unpack_rank_height(st);
            const RankDelta next_rd=rd+b300_rank_delta_step(N,p,H);
            const int next_h=b300_rank_height_advance(H,N);
            const MateID d=minsert(b,p-1,N);
            const Code base_rank=b300_sub_rank_delta(i,next_rd);

            // One lane per physical position. Build the Motzkin height prefix in
            // lane order (low physical bit -> high physical bit). All 32 lanes
            // participate so the shuffle masks remain uniform even though W<=28.
            const MateValue sv=lane<unsigned(TARGET_W)?mget(d,int(lane)):N;
            int step=(sv==L)-(sv==R);
            int pref=step;
#pragma unroll
            for(int off=1;off<32;off<<=1){
                const int x=__shfl_up_sync(mask,pref,off);
                if(int(lane)>=off)pref+=x;
            }
            const int pref_prev=lane?__shfl_up_sync(mask,pref,1):0;
            const int pref_pm2=__shfl_sync(mask,pref,p-2);
            const int pref_p=__shfl_sync(mask,pref,p);

            // Serial left scan stops at the first prefix below H. A ballot of
            // those barrier positions lets every candidate lane test that rule
            // in O(1), while before_left==0 is exactly the serial bal==0 test.
            const int before_left=pref_pm2-pref;
            const int after_left=pref_pm2-pref_prev;
            const bool left_pos=int(lane)<p-1;
            const unsigned left_barriers=__ballot_sync(mask,left_pos&&after_left<0);
            const unsigned above_lane=lane>=31?0u:(0xffffffffu<<(lane+1));
            const bool left_candidate=left_pos&&sv==L&&before_left==0&&((left_barriers&above_lane)==0);

            // Symmetric reverse-direction scan on the high side.
            const int before_right=-(pref_prev-pref_p);
            const int after_right=-(pref-pref_p);
            const bool right_pos=int(lane)>p&&lane<unsigned(TARGET_W);
            const unsigned right_barriers=__ballot_sync(mask,right_pos&&after_right<0);
            const unsigned below_lane=lane?((uint32_t(1)<<lane)-1u):0u;
            const bool right_candidate=right_pos&&sv==R&&before_right==0&&((right_barriers&below_lane)==0);

            // Prefix-scan the incremental grouped-rank corrections. This is the
            // algebraic parallel form of the old lane-0 ldelta/rsuffix loops.
            BlockClosureDelta left_diff=0,right_diff=0;
            if(left_pos&&(sv==R||sv==L)){
                const int hb=H+before_left;
                left_diff=BlockClosureDelta(block_pull_rank_contrib(sv,int(lane),hb+2))-
                          BlockClosureDelta(block_pull_rank_contrib(sv,int(lane),hb));
            }
            if(right_pos&&(sv==R||sv==L)){
                const int hq=H+after_right;
                right_diff=BlockClosureDelta(block_pull_rank_contrib(sv,int(lane),hq+2))-
                           BlockClosureDelta(block_pull_rank_contrib(sv,int(lane),hq));
            }
            BlockClosureDelta left_prefix=left_diff,right_prefix=right_diff;
#pragma unroll
            for(int off=1;off<32;off<<=1){
                const BlockClosureDelta lx=__shfl_up_sync(mask,left_prefix,off);
                const BlockClosureDelta rx=__shfl_up_sync(mask,right_prefix,off);
                if(int(lane)>=off){left_prefix+=lx;right_prefix+=rx;}
            }
            const BlockClosureDelta left_at_pm2=__shfl_sync(mask,left_prefix,p-2);
            const BlockClosureDelta right_at_p=__shfl_sync(mask,right_prefix,p);
            const BlockClosureDelta right_prefix_prev=lane?__shfl_up_sync(mask,right_prefix,1):BlockClosureDelta(0);

            bool has=false;Code source_rank=0;
            // Lane p can never be an LL/RR candidate because d[p] is N, so use
            // it for the RL source and keep one HBM request per participating lane.
            if(int(lane)==p&&H>0){
                const BlockClosureDelta x=BlockClosureDelta(block_pull_rank_contrib(R,p,H))+
                                          BlockClosureDelta(block_pull_rank_contrib(L,p-1,H-1));
                source_rank=block_pull_apply_delta(base_rank,x);has=true;
            }else if(left_candidate){
                const int q=int(lane),hb=H+before_left;
                const BlockClosureDelta l0=BlockClosureDelta(block_pull_rank_contrib(L,p,H))+
                                           BlockClosureDelta(block_pull_rank_contrib(L,p-1,H+1));
                const BlockClosureDelta span=left_at_pm2-left_prefix;
                const BlockClosureDelta x=l0+span+
                    BlockClosureDelta(block_pull_rank_contrib(R,q,hb+2))-
                    BlockClosureDelta(block_pull_rank_contrib(L,q,hb));
                source_rank=block_pull_apply_delta(base_rank,x);has=true;
            }else if(right_candidate){
                const int q=int(lane),hq=H+after_right;
                const BlockClosureDelta r0=BlockClosureDelta(block_pull_rank_contrib(R,p,H+2))+
                                           BlockClosureDelta(block_pull_rank_contrib(R,p-1,H+1));
                const BlockClosureDelta span=right_prefix_prev-right_at_p;
                const BlockClosureDelta x=r0+span+
                    BlockClosureDelta(block_pull_rank_contrib(L,q,hq))-
                    BlockClosureDelta(block_pull_rank_contrib(R,q,hq));
                source_rank=block_pull_apply_delta(base_rank,x);has=true;
            }

            // All closure source loads are now issued by their candidate lanes,
            // rather than serially after lane 0 has generated an index list.
            Count value=has?in_main[source_rank]:Count(0);
#pragma unroll
            for(int off=16;off;off>>=1){
                const Count other=__shfl_down_sync(mask,value,off);
                if(int(lane)<off)pull_add_mod(value,other);
            }
            if(lane==0){out_block[i]=value;rank_state[i]=b300_pack_rank_state(next_rd,next_h);}
            closure_mask&=closure_mask-1u;
        }
    }
}'''

s=replace_function(s,'b300_block_closure_warp_kernel',new_kernel)
# The old implementation needed 8 KiB/block of shared rank staging and lane-0
# serial rank generation. Both must disappear from this variant.
wp=s.find('b300_block_closure_warp_kernel(');wend=s.find('\n\nstatic Code rank_full',wp)
warp=s[wp:wend if wend>=0 else len(s)]
for stale in ('__shared__ Code ranks','rank_generation_lane0','ranks[warp_in_block]'):
    if stale in warp:raise SystemExit(f'stale serial closure artifact remains: {stale}')
for req in (
    'left_barriers=__ballot_sync','right_barriers=__ballot_sync',
    '__shfl_up_sync(mask,left_prefix','__shfl_up_sync(mask,right_prefix',
    'left_candidate','right_candidate','has?in_main[source_rank]',
    'lane==0){out_block[i]=value','closure_mask&=closure_mask-1u'
):
    if req not in warp:raise SystemExit(f'missing parallel closure artifact: {req}')
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(s)
print(f'generated {out} from {src}: b300_block_closure_warp_parallel_prefix=1 rank_generation_lane0=0 shared_rank_staging_bytes=0 physical_lanes=32 balance_prefix_scan=warp candidate_barriers=ballot rank_delta_prefix_scan=warp source_load_lane_parallel=1 proof=b300-block-closure-warp-parallel-prefix-proof')
