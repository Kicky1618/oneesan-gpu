#!/usr/bin/env python3
from __future__ import annotations
import pathlib,sys

if len(sys.argv)!=3:
    raise SystemExit('usage: gen-b300-block-closure-warpscan.py INPUT.cu OUTPUT.cu')
src=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);s=src.read_text()
for req in (
    'b300_block_closure_warp_kernel','b300_closure_warp_endpoint_masks',
    'block_pull_rank_shift2','block_pull_rank_cross_ll','block_pull_rank_cross_rr',
    'b300_pack_rank_state','b300_unpack_rank_delta','b300_unpack_rank_height'
):
    if req not in s:raise SystemExit(f'closure warpscan requires artifact: {req}')
use_cg='b300_rankstate_random_load_cg' in s
load=lambda expr: f'b300_rankstate_random_load_cg({expr})' if use_cg else f'*({expr})'

def replace_function(text:str,name:str,new:str)->str:
    p=text.find(name+'(')
    if p<0:raise SystemExit(f'function not found: {name}')
    start=text.rfind('\n',0,p)+1;brace=text.find('{',p);depth=0;end=-1
    for k in range(brace,len(text)):
        if text[k]=='{':depth+=1
        elif text[k]=='}':
            depth-=1
            if depth==0:end=k+1;break
    if end<0:raise SystemExit(f'function end not found: {name}')
    return text[:start]+new+text[end:]

new=r'''__global__ void b300_block_closure_warp_kernel(
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
            const Code base_rank=b300_sub_rank_delta(i,next_rd);
            const MateID d=minsert(b,p-1,N);
            const B300ClosureWarpEndpointMasks ep=b300_closure_warp_endpoint_masks(d);

            const int ql=p-2-int(lane);
            const bool vlane=ql>=0;
            const uint32_t qlb=vlane?(uint32_t(1)<<ql):0u;
            const MateValue lv=!vlane?N:((ep.l&qlb)?L:((ep.r&qlb)?R:N));
            const int lstep=(lv==L)-(lv==R);
            int lbal=lstep;
#pragma unroll
            for(int off=1;off<32;off<<=1){const int x=__shfl_up_sync(mask,lbal,off);if(int(lane)>=off)lbal+=x;}
            const int lpre=lbal-lstep;
            int lmin=lbal;
#pragma unroll
            for(int off=1;off<32;off<<=1){const int x=__shfl_up_sync(mask,lmin,off);if(int(lane)>=off)lmin=x<lmin?x:lmin;}
            const int lhb=H+lpre;
            BlockClosureDelta lterm=0;
            if(vlane&&lhb>=0&&lhb<=MAXW+1)lterm=block_pull_rank_shift2(lv,ql,lhb);
            BlockClosureDelta lpref=lterm;
#pragma unroll
            for(int off=1;off<32;off<<=1){const BlockClosureDelta x=__shfl_up_sync(mask,lpref,off);if(int(lane)>=off)lpref+=x;}
            const BlockClosureDelta lbefore=lpref-lterm;
            BlockClosureDelta linit=0;
            if(lane==0)linit=BlockClosureDelta(block_pull_rank_contrib(L,p,H))+BlockClosureDelta(block_pull_rank_contrib(L,p-1,H+1));
            linit=__shfl_sync(mask,linit,0);
            const bool lcand=vlane&&lv==L&&lpre==0&&lmin>=0;
            const Code lrank=lcand?block_pull_apply_delta(base_rank,linit+lbefore+block_pull_rank_cross_ll(ql,lhb)):Code(0);

            const int qr=p+1+int(lane);
            const bool rlane=qr<TARGET_W;
            const uint32_t qrb=rlane?(uint32_t(1)<<qr):0u;
            const MateValue rv=!rlane?N:((ep.l&qrb)?L:((ep.r&qrb)?R:N));
            const int rstep=(rv==R)-(rv==L);
            int rbal=rstep;
#pragma unroll
            for(int off=1;off<32;off<<=1){const int x=__shfl_up_sync(mask,rbal,off);if(int(lane)>=off)rbal+=x;}
            const int rpre=rbal-rstep;
            int rmin=rbal;
#pragma unroll
            for(int off=1;off<32;off<<=1){const int x=__shfl_up_sync(mask,rmin,off);if(int(lane)>=off)rmin=x<rmin?x:rmin;}
            const int rhq=H+rbal;
            BlockClosureDelta rterm=0;
            if(rlane&&rhq>=0&&rhq<=MAXW+1)rterm=block_pull_rank_shift2(rv,qr,rhq);
            BlockClosureDelta rpref=rterm;
#pragma unroll
            for(int off=1;off<32;off<<=1){const BlockClosureDelta x=__shfl_up_sync(mask,rpref,off);if(int(lane)>=off)rpref+=x;}
            const BlockClosureDelta rbefore=rpref-rterm;
            BlockClosureDelta rinit=0;
            if(lane==0)rinit=BlockClosureDelta(block_pull_rank_contrib(R,p,H+2))+BlockClosureDelta(block_pull_rank_contrib(R,p-1,H+1));
            rinit=__shfl_sync(mask,rinit,0);
            const bool rcand=rlane&&rv==R&&rpre==0&&rmin>=0;
            const Code rrank=rcand?block_pull_apply_delta(base_rank,rinit+rbefore+block_pull_rank_cross_rr(qr,rhq)):Code(0);

            Code rlrank=0;bool rlcand=false;
            if(lane==0&&H>0){
                const BlockClosureDelta x=BlockClosureDelta(block_pull_rank_contrib(R,p,H))+BlockClosureDelta(block_pull_rank_contrib(L,p-1,H-1));
                rlrank=block_pull_apply_delta(base_rank,x);rlcand=true;
            }

            Count sum=0;
            if(lcand)pull_add_mod(sum,''' + load('in_main+lrank') + r''');
            if(rcand)pull_add_mod(sum,''' + load('in_main+rrank') + r''');
            if(rlcand)pull_add_mod(sum,''' + load('in_main+rlrank') + r''');
#pragma unroll
            for(int off=16;off;off>>=1){const Count x=__shfl_down_sync(mask,sum,off);if(lane<unsigned(off))pull_add_mod(sum,x);}
            if(lane==0){out_block[i]=sum;rank_state[i]=b300_pack_rank_state(next_rd,next_h);}
            __syncwarp(mask);
            closure_mask&=closure_mask-1u;
        }
    }
}'''
s=replace_function(s,'b300_block_closure_warp_kernel',new)

for stale in ('__shared__ Code ranks[MAX_WARPS][MAX_TERMS]','ranks[warp_in_block]','rank_generation_lane0'):
    if stale in s:raise SystemExit(f'stale serial closure artifact remains: {stale}')
for req in ('const int ql=p-2-int(lane)','const int qr=p+1+int(lane)','__shfl_up_sync','lpre==0&&lmin>=0','rpre==0&&rmin>=0','block_pull_rank_cross_ll','block_pull_rank_cross_rr','rank_state[i]=b300_pack_rank_state(next_rd,next_h)'):
    if req not in s:raise SystemExit(f'missing closure warpscan artifact: {req}')
if use_cg:
    for req in ('b300_rankstate_random_load_cg(in_main+lrank)','b300_rankstate_random_load_cg(in_main+rrank)','b300_rankstate_random_load_cg(in_main+rlrank)'):
        if req not in s:raise SystemExit(f'missing closure warpscan CG composition: {req}')
else:
    for req in ('*(in_main+lrank)','*(in_main+rrank)','*(in_main+rlrank)'):
        if req not in s:raise SystemExit(f'missing closure warpscan normal load: {req}')

out.parent.mkdir(parents=True,exist_ok=True);out.write_text(s)
print(f'generated {out} from {src}: b300_block_closure_warpscan=1 candidate_rank_generation=warp_prefix_scan lane0_serial_rank_generation=0 shared_rank_queue_bytes=0 left_positions_parallel=32 right_positions_parallel=32 random_loads_parallel=LL+RR+RL prefix_balance_exact=1 prefix_min_break_exact=1 random_load_cg={int(use_cg)}')
