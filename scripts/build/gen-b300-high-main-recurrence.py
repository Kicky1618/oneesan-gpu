#!/usr/bin/env python3
from __future__ import annotations
import pathlib,sys

if len(sys.argv)!=3:raise SystemExit('usage: gen-b300-high-main-recurrence.py INPUT.cu OUTPUT.cu')
src=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);s=src.read_text()
for req in ('main_pull_kernel_ilp2','b300_main_pull_prepare','b300_low_window_cache_active','b300_low_cached_drop_rank','b300_pack_low_window_main_mate'):
    if req not in s:raise SystemExit(f'high-main recurrence requires artifact: {req}')

def replace_function(text:str,name:str,new:str)->str:
    p=text.find(name+'(')
    if p<0:raise SystemExit(f'function not found: {name}')
    start=text.rfind('\n',0,p)+1;brace=text.find('{',p)
    if brace<0:raise SystemExit(f'function brace not found: {name}')
    depth=0;end=-1
    for i in range(brace,len(text)):
        if text[i]=='{':depth+=1
        elif text[i]=='}':
            depth-=1
            if depth==0:end=i+1;break
    if end<0:raise SystemExit(f'function end not found: {name}')
    return text[:start]+new+text[end:]

# Rewrite only the two production materialization calls before inserting a helper
# which itself intentionally calls the low-window packer.
pack_call='b300_pack_low_window_main_mate(m)'
if s.count(pack_call)!=2:raise SystemExit(f'expected two main mate packing calls before helper insertion, got {s.count(pack_call)}')
s=s.replace(pack_call,'b300_pack_main_transition_cache(m)',2)

high_rank='b300_high_chunk_drop_rank(i,m,p)' if 'b300_high_chunk_drop_rank' in s else 'rank_drop_n_t<TARGET_W>(i,m,p)'
# gather_main/materialize kernels call b300_pack_main_transition_cache, so all
# helper definitions must appear before the first gather_main kernel definition.
marker='__global__ void gather_main_kernel('
if s.count(marker)!=1:raise SystemExit(f'gather_main marker expected once, got {s.count(marker)}')
helper=r'''
__device__ __forceinline__ bool b300_high_main_state_active(){
    if constexpr(TARGET_W!=28)return false;
    else {
        constexpr uint32_t HIGH=((uint32_t(1)<<28)-1u)^((uint32_t(1)<<14)-1u);
        return (D_MAIN_FIXED&HIGH)==0;
    }
}
__device__ __forceinline__ uint32_t b300_high_trit_chunk(MateID m,int base){
    const uint32_t a=base<=27?uint32_t(mget(m,base)):0u;
    const uint32_t b=base+1<=27?uint32_t(mget(m,base+1)):0u;
    const uint32_t c=base+2<=27?uint32_t(mget(m,base+2)):0u;
    return a+3u*b+9u*c;
}
__device__ __forceinline__ MateValue b300_high_state_get(MateID x,int p){
    const int q=p-14,c=q/3,r=q-c*3;const uint32_t z=uint32_t((x>>(5*c))&31ULL);
    uint32_t v;if(r==0){uint32_t a=z/3;v=z-a*3;}else if(r==1){uint32_t a=z/3;v=a-(a/3)*3;}else v=z/9;
    return MateValue(v);
}
__device__ __forceinline__ MateValuePair b300_high_state_pair(MateID x,int p){
    return MateValuePair(uint32_t(b300_high_state_get(x,p-1))|(uint32_t(b300_high_state_get(x,p))<<2));
}
__device__ __forceinline__ long long b300_high_state_delta(MateID x){
    const unsigned long long u=(x>>25)&((MateID(1)<<35)-1ULL);return static_cast<long long>(u<<29)>>29;
}
__device__ __forceinline__ int b300_high_state_height(MateID x){return int((x>>60)&15ULL);}
__device__ __forceinline__ MateID b300_high_state_pack_current(MateID x,long long d,int h){
    constexpr MateID TM=(MateID(1)<<25)-1ULL,DM=(MateID(1)<<35)-1ULL;
    return (x&TM)|((MateID(d)&DM)<<25)|(MateID(h&15)<<60);
}
__device__ __forceinline__ MateID b300_pack_high_main_state(MateID m){
    MateID trits=0;
#pragma unroll
    for(int c=0;c<5;++c)trits|=MateID(b300_high_trit_chunk(m,14+3*c))<<(5*c);
    return b300_high_state_pack_current(trits,0,1);
}
__device__ __forceinline__ MateID b300_pack_main_transition_cache(MateID m){
    if(b300_high_main_state_active())return b300_pack_high_main_state(m);
    return b300_pack_low_window_main_mate(m);
}
__device__ __forceinline__ void b300_high_state_step(long long&d,int&h,int p,MateValue v){
    if(v==R){d+=static_cast<long long>(D_BLOCK_DP[p-1][h])-static_cast<long long>(D_MAIN_DP[p][h]);--h;}
    else if(v==L){const Code b=D_BLOCK_DP[p-1][h]+(h?D_BLOCK_DP[p-1][h-1]:0),a=D_MAIN_DP[p][h]+(h?D_MAIN_DP[p][h-1]:0);d+=static_cast<long long>(b)-static_cast<long long>(a);++h;}
}
__device__ __forceinline__ MateID b300_high_state_advance(MateID x,int p){
    long long d=b300_high_state_delta(x);int h=b300_high_state_height(x);b300_high_state_step(d,h,p,b300_high_state_get(x,p));return b300_high_state_pack_current(x,d,h);
}
__device__ __forceinline__ Code b300_high_state_drop_rank(Code main_rank,MateID x){
    const long long d=b300_high_state_delta(x);return d>=0?main_rank+Code(d):main_rank-Code(-d);
}
__device__ __forceinline__ Code b300_main_pair_rank_h(Code dst_rank,int p,MateValuePair pair,int h){
    switch(pair){
        case LR:{const Code d=D_MAIN_DP[p][h]+(h?D_MAIN_DP[p][h-1]:0)+D_MAIN_DP[p-1][h+1];return dst_rank-d;}
        case NR:{const Code d=D_MAIN_DP[p][h]-D_MAIN_DP[p-1][h];return dst_rank+d;}
        case NL:{const Code a=D_MAIN_DP[p][h]+(h?D_MAIN_DP[p][h-1]:0),b=D_MAIN_DP[p-1][h]+(h?D_MAIN_DP[p-1][h-1]:0);return dst_rank+(a-b);}
        default:return dst_rank;
    }
}
'''
s=s.replace(marker,helper+'\n'+marker,1)

prepare=f'''__device__ __forceinline__ void b300_main_pull_prepare(
    Code i,MateID m,int p,Code nblock,
    Code& pair_j,bool& has_pair,Code& block_j,bool& has_block
){{
    const bool hs=b300_high_main_state_active();
    const MateValuePair pair=hs?b300_high_state_pair(m,p):mpair(m,p);
    has_pair=(pair==LR||pair==NR||pair==NL);
    if(has_pair)pair_j=hs?b300_main_pair_rank_h(i,p,pair,b300_high_state_height(m)):main_pull_direct_pair_source_rank(i,m,p);
    has_block=false;
    const MateValue vp=hs?b300_high_state_get(m,p):mget(m,p);
    if(nblock&&vp==N){{
        block_j=b300_low_window_cache_active()?b300_low_cached_drop_rank(i,m,p):(hs?b300_high_state_drop_rank(i,m):{high_rank});
        has_block=block_j<nblock;
    }}
}}'''
s=replace_function(s,'b300_main_pull_prepare',prepare)

old='const Count* __restrict__ in,const MateID* __restrict__ mates,Code n,'
if s.count(old)!=1:raise SystemExit(f'ILP2 mate pointer anchor count={s.count(old)}')
s=s.replace(old,'const Count* __restrict__ in,MateID* __restrict__ mates,Code n,',1)
old='uint64_t a0=uint64_t(self0)+pair0+block0;if(a0>=mod)a0-=mod;if(a0>=mod)a0-=mod;out_main[i0]=Count(a0);'
if s.count(old)!=1:raise SystemExit('ILP2 lane0 store anchor not unique')
s=s.replace(old,old+'if(b300_high_main_state_active())mates[i0]=b300_high_state_advance(m0,p);',1)
old='if(v1){uint64_t a1=uint64_t(self1)+pair1+block1;if(a1>=mod)a1-=mod;if(a1>=mod)a1-=mod;out_main[i1]=Count(a1);}'
if s.count(old)!=1:raise SystemExit('ILP2 lane1 store anchor not unique')
s=s.replace(old,'if(v1){uint64_t a1=uint64_t(self1)+pair1+block1;if(a1>=mod)a1-=mod;if(a1>=mod)a1-=mod;out_main[i1]=Count(a1);if(b300_high_main_state_active())mates[i1]=b300_high_state_advance(m1,p);}',1)

for required in ('b300_high_main_state_active','b300_high_state_drop_rank','b300_main_pair_rank_h','b300_pack_main_transition_cache','b300_high_state_advance(m1,p)','return b300_pack_low_window_main_mate(m);'):
    if required not in s:raise SystemExit(f'missing high-main recurrence artifact: {required}')
if s.count('b300_pack_main_transition_cache(m)')!=2:raise SystemExit(f'expected exactly two production transition-cache packing calls, got {s.count("b300_pack_main_transition_cache(m)")}')
if s.find('b300_pack_main_transition_cache',s.find('__global__ void gather_main_kernel'))>=0 and s.find('b300_pack_main_transition_cache')>s.find('__global__ void gather_main_kernel'):
    raise SystemExit('transition-cache helper was emitted after gather_main')
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(s)
print(f'generated {out} from {src}: high_main_recurrence=1 ilp=2 extra_state_bytes=0 trit_bits=25 signed_delta_bits=35 height_bits=4 helper_before_gather=1 mate_hbm_store_per_state_step=8 high_drop_walk_or_table_loads_per_state_step=0 high_height_walk_per_state_step=0')
