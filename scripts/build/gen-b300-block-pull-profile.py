#!/usr/bin/env python3
from __future__ import annotations
import pathlib,sys

if len(sys.argv)!=3:
    raise SystemExit('usage: gen-b300-block-pull-profile.py INPUT.cu OUTPUT.cu')
src=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);s=src.read_text()
for req in (
    'template<bool CACHED_BLOCK_MATE>\n__global__ void block_pull_kernel',
    'while(left){',
    'while(right){',
    'out_block[i]=acc;',
    'for(size_t ri=0;ri<mods.size();++ri){Count mod=mods[ri];',
    'backend=gridfp-b300-hbm32-forced2window-opt-batch',
):
    if req not in s:raise SystemExit(f'block pull profiler requires artifact: {req}')

def once(old:str,new:str,label:str)->None:
    global s
    n=s.count(old)
    if n!=1:raise SystemExit(f'{label}: expected one match got {n}')
    s=s.replace(old,new,1)

marker='template<bool CACHED_BLOCK_MATE>\n__global__ void block_pull_kernel'
prof=r'''#ifndef B300_BLOCK_PROF_LOG2
#define B300_BLOCK_PROF_LOG2 20
#endif
static_assert(B300_BLOCK_PROF_LOG2>=8 && B300_BLOCK_PROF_LOG2<=30,"B300_BLOCK_PROF_LOG2 must be 8..30");
enum B300BlockProfMetric:int{
    B300_BP_SAMPLED=0,
    B300_BP_LOOK_ENDPOINT=1,
    B300_BP_LOOK_N=2,
    B300_BP_HPOS=3,
    B300_BP_LEFT_ITERS=4,
    B300_BP_RIGHT_ITERS=5,
    B300_BP_LEFT_CAND=6,
    B300_BP_RIGHT_CAND=7,
    B300_BP_ENDPOINTS=8,
    B300_BP_METRICS=9
};
__device__ unsigned long long B300_BLOCK_PROF[2][B300_BP_METRICS];
__device__ __forceinline__ bool b300_block_prof_sample(Code i,int p){
    uint32_t x=uint32_t(i)^uint32_t(i>>32)^(D_BLOCK_OCC*0x9e3779b9u)^(uint32_t(p)*0x85ebca6bu);
    x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;x^=x>>16;
    return (x&((uint32_t(1)<<B300_BLOCK_PROF_LOG2)-1u))==0;
}
__device__ __forceinline__ void b300_block_prof_add(int w,int metric,unsigned long long v){
    if(v)atomicAdd(&B300_BLOCK_PROF[w][metric],v);
}

'''
once(marker,prof+marker,'profile declarations')

once(
'''        Count acc=0;
        MateValue look=mget(b,p-1);''',
'''        Count acc=0;
        const bool b300_prof=b300_block_prof_sample(i,p);
        const int b300_prof_w=p>=15?0:1;
        unsigned long long b300_prof_left=0,b300_prof_right=0,b300_prof_lcand=0,b300_prof_rcand=0,b300_prof_endpoints=0;
        MateValue look=mget(b,p-1);
        if(b300_prof){
            b300_block_prof_add(b300_prof_w,B300_BP_SAMPLED,1);
            if(look==R||look==L)b300_block_prof_add(b300_prof_w,B300_BP_LOOK_ENDPOINT,1);
            else if(look==N)b300_block_prof_add(b300_prof_w,B300_BP_LOOK_N,1);
        }''',
'profile state prologue')

once(
'''            if(H>0){
                const BlockClosureDelta rd=''',
'''            if(H>0){
                if(b300_prof)b300_block_prof_add(b300_prof_w,B300_BP_HPOS,1);
                const BlockClosureDelta rd=''',
'profile H positive')

once(
'''            const uint32_t endpoints=block_pull_endpoint_mask(d);''',
'''            const uint32_t endpoints=block_pull_endpoint_mask(d);
            if(b300_prof)b300_prof_endpoints=__popc(endpoints);''',
'profile endpoint popcount')

once(
'''            while(left){
                const int q=31-__clz(left);const MateValue v=mget(d,q);''',
'''            while(left){
                if(b300_prof)++b300_prof_left;
                const int q=31-__clz(left);const MateValue v=mget(d,q);''',
'profile left iterations')
once(
'''                if(bal==0&&v==L){
                    const BlockClosureDelta xd=ldelta+''',
'''                if(bal==0&&v==L){
                    if(b300_prof)++b300_prof_lcand;
                    const BlockClosureDelta xd=ldelta+''',
'profile left candidates')

once(
'''            while(right){
                const int q=__ffs(right)-1;const MateValue v=mget(d,q);''',
'''            while(right){
                if(b300_prof)++b300_prof_right;
                const int q=__ffs(right)-1;const MateValue v=mget(d,q);''',
'profile right iterations')
once(
'''                if(bal==0&&v==R){
                    const BlockClosureDelta xd=''',
'''                if(bal==0&&v==R){
                    if(b300_prof)++b300_prof_rcand;
                    const BlockClosureDelta xd=''',
'profile right candidates')

once(
'''        out_block[i]=acc;''',
'''        if(b300_prof){
            b300_block_prof_add(b300_prof_w,B300_BP_LEFT_ITERS,b300_prof_left);
            b300_block_prof_add(b300_prof_w,B300_BP_RIGHT_ITERS,b300_prof_right);
            b300_block_prof_add(b300_prof_w,B300_BP_LEFT_CAND,b300_prof_lcand);
            b300_block_prof_add(b300_prof_w,B300_BP_RIGHT_CAND,b300_prof_rcand);
            b300_block_prof_add(b300_prof_w,B300_BP_ENDPOINTS,b300_prof_endpoints);
        }
        out_block[i]=acc;''',
'profile state flush')

res='for(size_t ri=0;ri<mods.size();++ri){Count mod=mods[ri];'
once(res,res+r'''
        for(int d=0;d<ng;++d){
            ck(cudaSetDevice(d),"set block profile reset device");
            unsigned long long z[2][B300_BP_METRICS]{};
            ck(cudaMemcpyToSymbol(B300_BLOCK_PROF,z,sizeof(z)),"reset block profile");
        }''','profile residue reset')

backend=s.find('std::cout<<"backend=gridfp-b300-hbm32-forced2window-opt-batch n="')
if backend<0:raise SystemExit('backend output statement not found')
end=s.find('std::endl;',backend)
if end<0:raise SystemExit('backend output statement end not found')
end+=len('std::endl;')
collect=r'''
        unsigned long long b300_prof_sum[2][B300_BP_METRICS]{};
        for(int d=0;d<ng;++d){
            ck(cudaSetDevice(d),"set block profile read device");
            unsigned long long q[2][B300_BP_METRICS]{};
            ck(cudaMemcpyFromSymbol(q,B300_BLOCK_PROF,sizeof(q)),"read block profile");
            for(int w=0;w<2;++w)for(int k=0;k<B300_BP_METRICS;++k)b300_prof_sum[w][k]+=q[w][k];
        }
        std::cout<<"b300_block_profile modulus="<<mod<<" sample_log2="<<B300_BLOCK_PROF_LOG2
                 <<" high_sampled="<<b300_prof_sum[0][B300_BP_SAMPLED]
                 <<" high_look_endpoint="<<b300_prof_sum[0][B300_BP_LOOK_ENDPOINT]
                 <<" high_look_n="<<b300_prof_sum[0][B300_BP_LOOK_N]
                 <<" high_hpos="<<b300_prof_sum[0][B300_BP_HPOS]
                 <<" high_left_iters="<<b300_prof_sum[0][B300_BP_LEFT_ITERS]
                 <<" high_right_iters="<<b300_prof_sum[0][B300_BP_RIGHT_ITERS]
                 <<" high_left_candidates="<<b300_prof_sum[0][B300_BP_LEFT_CAND]
                 <<" high_right_candidates="<<b300_prof_sum[0][B300_BP_RIGHT_CAND]
                 <<" high_endpoint_popcnt="<<b300_prof_sum[0][B300_BP_ENDPOINTS]
                 <<" low_sampled="<<b300_prof_sum[1][B300_BP_SAMPLED]
                 <<" low_look_endpoint="<<b300_prof_sum[1][B300_BP_LOOK_ENDPOINT]
                 <<" low_look_n="<<b300_prof_sum[1][B300_BP_LOOK_N]
                 <<" low_hpos="<<b300_prof_sum[1][B300_BP_HPOS]
                 <<" low_left_iters="<<b300_prof_sum[1][B300_BP_LEFT_ITERS]
                 <<" low_right_iters="<<b300_prof_sum[1][B300_BP_RIGHT_ITERS]
                 <<" low_left_candidates="<<b300_prof_sum[1][B300_BP_LEFT_CAND]
                 <<" low_right_candidates="<<b300_prof_sum[1][B300_BP_RIGHT_CAND]
                 <<" low_endpoint_popcnt="<<b300_prof_sum[1][B300_BP_ENDPOINTS]
                 <<std::endl;
'''
s=s[:end]+collect+s[end:]

for req in ('B300_BLOCK_PROF[2][B300_BP_METRICS]','b300_block_prof_sample','high_left_iters=','low_right_candidates='):
    if req not in s:raise SystemExit(f'profile artifact missing after transform: {req}')
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(s)
print(f'generated {out} from {src}: block_pull_profile=1 sample_log2_macro=B300_BLOCK_PROF_LOG2 windows=high15_27,low2_14 metrics=sampled,look_endpoint,look_n,hpos,left_iters,right_iters,left_candidates,right_candidates,endpoint_popcnt production_default=off')
