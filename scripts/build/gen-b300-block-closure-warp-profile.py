#!/usr/bin/env python3
from __future__ import annotations
import pathlib,sys

if len(sys.argv)!=3:
    raise SystemExit('usage: gen-b300-block-closure-warp-profile.py INPUT.cu OUTPUT.cu')
src=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);s=src.read_text()
for req in (
    'b300_block_closure_warp_kernel','__ballot_sync','__shfl_down_sync',
    'cnt=__shfl_sync(mask,cnt,0);','static Code rank_full(MateID m,int width)',
    'std::cout<<"backend=gridfp-b300-hbm32'
):
    if req not in s:raise SystemExit(f'closure-warp profiler requires artifact: {req}')

# Per-device counters are local device symbols. 29 p values x 29 endpoint counts
# x 33 valid-source counts costs ~217 KiB/GPU, negligible versus n=27 state HBM.
marker='\n\n// Each warp first classifies 32 blocked states in parallel.'
if marker not in s:raise SystemExit('closure warp marker missing')
prof=r'''

#ifndef B300_CLOSURE_WARP_PROF_LOG2
#define B300_CLOSURE_WARP_PROF_LOG2 20
#endif
static_assert(B300_CLOSURE_WARP_PROF_LOG2>=8 && B300_CLOSURE_WARP_PROF_LOG2<=30,
              "B300_CLOSURE_WARP_PROF_LOG2 must be 8..30");
__device__ unsigned long long B300_CLOSURE_WARP_PROF[29][29][33];
__device__ __forceinline__ bool b300_closure_warp_prof_sample(Code i,int p){
    uint32_t x=uint32_t(i)^uint32_t(i>>32)^(D_BLOCK_OCC*0x9e3779b9u)^(uint32_t(p)*0x85ebca6bu);
    x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;x^=x>>16;
    return (x&((uint32_t(1)<<B300_CLOSURE_WARP_PROF_LOG2)-1u))==0;
}
__device__ __forceinline__ int b300_closure_warp_prof_endpoints(MateID b){
    constexpr unsigned long long EVEN=0x5555555555555555ULL;
    return __popcll((static_cast<unsigned long long>(b)|(static_cast<unsigned long long>(b)>>1))&EVEN);
}
'''
s=s.replace(marker,prof+marker,1)

# Profile exactly once per closure state, after lane0 has generated the valid
# source-rank list and before cnt is broadcast. No extra closure walk is added.
anchor='''            cnt=__shfl_sync(mask,cnt,0);'''
if s.count(anchor)!=1:raise SystemExit(f'cnt broadcast anchor expected once got {s.count(anchor)}')
repl=r'''            if(lane==0&&b300_closure_warp_prof_sample(i,p)){
                const int ep=b300_closure_warp_prof_endpoints(b);
                if(p>=0&&p<29&&ep>=0&&ep<29&&cnt>=0&&cnt<33)atomicAdd(&B300_CLOSURE_WARP_PROF[p][ep][cnt],1ULL);
            }
            cnt=__shfl_sync(mask,cnt,0);'''
s=s.replace(anchor,repl,1)

# Reset every device before timed row processing. The generated program handles
# one modulus per invocation, so one reset is sufficient.
reset_anchor='''    auto wall0=std::chrono::steady_clock::now();int done_windows=0;'''
if s.count(reset_anchor)!=1:raise SystemExit(f'wall start anchor expected once got {s.count(reset_anchor)}')
reset=r'''    for(int d=0;d<ng;++d){
        ck(cudaSetDevice(d),"set closure warp profile reset device");
        unsigned long long z[29][29][33]{};
        ck(cudaMemcpyToSymbol(B300_CLOSURE_WARP_PROF,z,sizeof(z)),"reset closure warp profile");
    }
    auto wall0=std::chrono::steady_clock::now();int done_windows=0;'''
s=s.replace(reset_anchor,reset,1)

# Read all per-device symbols after the timed result is determined. Print compact
# endpoint and p summaries plus threshold work-retention curves rather than 27k bins.
backend=s.find('std::cout<<"backend=gridfp-b300-hbm32')
if backend<0:raise SystemExit('backend output statement missing')
stmt_end=s.find(';',backend)
if stmt_end<0:raise SystemExit('backend output statement end missing')
stmt_end+=1
collect=r'''
    unsigned long long b300_cwp[29][29][33]{};
    for(int d=0;d<ng;++d){
        ck(cudaSetDevice(d),"set closure warp profile read device");
        unsigned long long q[29][29][33]{};
        ck(cudaMemcpyFromSymbol(q,B300_CLOSURE_WARP_PROF,sizeof(q)),"read closure warp profile");
        for(int p=0;p<29;++p)for(int e=0;e<29;++e)for(int c=0;c<33;++c)b300_cwp[p][e][c]+=q[p][e][c];
    }
    unsigned long long b300_total_states=0,b300_total_candidates=0;
    for(int p=0;p<29;++p){
        unsigned long long ps=0,pe=0,pc=0;
        for(int e=0;e<29;++e)for(int c=0;c<33;++c){const auto z=b300_cwp[p][e][c];ps+=z;pe+=z*unsigned(e);pc+=z*unsigned(c);}
        b300_total_states+=ps;b300_total_candidates+=pc;
        if(ps)std::cout<<"b300_closure_warp_profile_p p="<<p<<" samples="<<ps<<" endpoint_sum="<<pe<<" candidate_sum="<<pc
                       <<" avg_endpoints="<<double(pe)/double(ps)<<" avg_candidates="<<double(pc)/double(ps)<<std::endl;
    }
    for(int e=0;e<29;++e){
        unsigned long long es=0,ec=0;
        for(int p=0;p<29;++p)for(int c=0;c<33;++c){const auto z=b300_cwp[p][e][c];es+=z;ec+=z*unsigned(c);}
        if(es)std::cout<<"b300_closure_warp_profile_endpoint endpoints="<<e<<" samples="<<es<<" candidate_sum="<<ec
                       <<" avg_candidates="<<double(ec)/double(es)<<std::endl;
    }
    for(int t=1;t<=20;++t){
        unsigned long long ws=0,wc=0;
        for(int p=0;p<29;++p)for(int e=t;e<29;++e)for(int c=0;c<33;++c){const auto z=b300_cwp[p][e][c];ws+=z;wc+=z*unsigned(c);}
        const double sf=b300_total_states?double(ws)/double(b300_total_states):0.0;
        const double cf=b300_total_candidates?double(wc)/double(b300_total_candidates):0.0;
        std::cout<<"b300_closure_warp_profile_threshold threshold="<<t<<" warp_state_fraction="<<sf<<" warp_candidate_fraction="<<cf
                 <<" scalar_state_fraction="<<(1.0-sf)<<" samples="<<b300_total_states<<" candidates="<<b300_total_candidates<<std::endl;
    }
    std::cout<<"b300_closure_warp_profile sample_log2="<<B300_CLOSURE_WARP_PROF_LOG2<<" samples="<<b300_total_states
             <<" candidates="<<b300_total_candidates<<" exact_output_unchanged=1"<<std::endl;
'''
s=s[:stmt_end]+collect+s[stmt_end:]

for req in (
    'B300_CLOSURE_WARP_PROF[29][29][33]','b300_closure_warp_prof_sample',
    'b300_closure_warp_profile_endpoint endpoints=','b300_closure_warp_profile_threshold threshold=',
    'cudaMemcpyFromSymbol(q,B300_CLOSURE_WARP_PROF'
):
    if req not in s:raise SystemExit(f'closure-warp profile artifact missing: {req}')
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(s)
print(f'generated {out} from {src}: b300_closure_warp_profile=1 sample_log2_macro=B300_CLOSURE_WARP_PROF_LOG2 dimensions=p,endpoint_count,valid_candidate_count profile_bytes_per_gpu={29*29*33*8} extra_closure_walks=0 production_default=off')
