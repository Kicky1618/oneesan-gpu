#!/usr/bin/env python3
from __future__ import annotations
import pathlib,sys

if len(sys.argv)!=3:raise SystemExit('usage: gen-b300-main-recurrence.py INPUT.cu OUTPUT.cu')
src=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);s=src.read_text()
for req in ('main_pull_kernel_ilp2','b300_main_pull_prepare','b300_low_window_cache_active','b300_low_cached_drop_rank','b300_pack_low_window_main_mate','main_pull_direct_pair_source_rank'):
    if req not in s:raise SystemExit(f'unified main recurrence requires artifact: {req}')

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

# Convert the existing low-window cache into a recurrent 64-bit state:
# 25 trit bits + signed31 current drop delta + 4-bit height + ready bit.
low_pack=r'''__device__ __forceinline__ uint32_t b300_main_trit3(MateID m,int base,int limit){
    const uint32_t a=base<limit?uint32_t(mget(m,base)):0u;
    const uint32_t b=base+1<limit?uint32_t(mget(m,base+1)):0u;
    const uint32_t c=base+2<limit?uint32_t(mget(m,base+2)):0u;
    return a+3u*b+9u*c;
}
__device__ __forceinline__ uint32_t b300_main_trit_get(MateID x,int q){
    const int c=q/3,r=q-c*3;const uint32_t z=uint32_t((x>>(5*c))&31ULL);
    if(r==0){const uint32_t a=z/3;return z-a*3;}
    if(r==1){const uint32_t a=z/3;return a-(a/3)*3;}
    return z/9;
}
__device__ __forceinline__ MateValue b300_low_state_get(MateID x,int p){return MateValue(b300_main_trit_get(x,p));}
__device__ __forceinline__ MateValuePair b300_low_state_pair(MateID x,int p){return MateValuePair(uint32_t(b300_low_state_get(x,p-1))|(uint32_t(b300_low_state_get(x,p))<<2));}
__device__ __forceinline__ long long b300_low_state_delta(MateID x){const unsigned long long u=(x>>25)&0x7fffffffULL;return static_cast<long long>(u<<33)>>33;}
__device__ __forceinline__ int b300_low_state_height(MateID x){return int((x>>56)&15ULL);}
__device__ __forceinline__ bool b300_low_state_ready(MateID x){return ((x>>60)&1ULL)!=0;}
__device__ __forceinline__ MateID b300_low_state_pack_current(MateID x,long long d,int h){constexpr MateID TM=(MateID(1)<<25)-1ULL,DM=(MateID(1)<<31)-1ULL;return (x&TM)|((MateID(d)&DM)<<25)|(MateID(h&15)<<56)|(MateID(1)<<60);}
__device__ __forceinline__ void b300_low_state_step(long long&d,int&h,int p,MateValue v){
    if(v==R){d+=static_cast<long long>(D_FULL_DP[p-1][h])-static_cast<long long>(D_FULL_DP[p][h]);--h;}
    else if(v==L){const Code b=D_FULL_DP[p-1][h]+(h?D_FULL_DP[p-1][h-1]:0),a=D_FULL_DP[p][h]+(h?D_FULL_DP[p][h-1]:0);d+=static_cast<long long>(b)-static_cast<long long>(a);++h;}
}
__device__ __forceinline__ MateID b300_low_state_advance(MateID x,int p){long long d=b300_low_state_delta(x);int h=b300_low_state_height(x);b300_low_state_step(d,h,p,b300_low_state_get(x,p));return b300_low_state_pack_current(x,d,h);}
__device__ __forceinline__ MateID b300_pack_low_window_main_mate(MateID m){
    if(!b300_low_window_cache_active())return m;
    MateID trits=0;
#pragma unroll
    for(int c=0;c<5;++c)trits|=MateID(b300_main_trit3(m,3*c,15))<<(5*c);
    constexpr int LOW=15;constexpr uint32_t CODE_MASK=(1u<<(2*13))-1u;
    const uint32_t code=uint32_t((m>>(2*LOW))&CODE_MASK),mmask=(D_MAIN_OCC>>LOW)&((1u<<13)-1u),bmask=(D_BLOCK_OCC>>(LOW-1))&((1u<<13)-1u);
    const uint32_t main_base=b300_high_base_lookup(code,mmask,false),block_base=b300_high_base_lookup(code,bmask,true),delta_mag=main_base-block_base;
    const MateID high=m>>(2*LOW);constexpr MateID EVEN=0x5555555555555555ULL;const int nr=__popcll(high&EVEN),nl=__popcll((high>>1)&EVEN);const uint32_t enter_h=uint32_t(1+nl-nr);
    return trits|(MateID(delta_mag)<<25)|(MateID(enter_h)<<56);
}'''
s=replace_function(s,'b300_pack_low_window_main_mate',low_pack)

low_height=r'''__device__ __forceinline__ int b300_low_cached_height_before(MateID x,int p){
    if(b300_low_state_ready(x))return b300_low_state_height(x);
    int h=b300_low_state_height(x);for(int pos=14;pos>p;--pos){const MateValue v=b300_low_state_get(x,pos);h+=(v==L)-(v==R);}return h;
}'''
s=replace_function(s,'b300_low_cached_height_before',low_height)
low_drop=r'''__device__ __forceinline__ Code b300_low_cached_drop_rank(Code main_rank,MateID x,int p){
    long long d;if(b300_low_state_ready(x))d=b300_low_state_delta(x);else{d=-static_cast<long long>((x>>25)&0x7fffffffULL);int h=b300_low_state_height(x);for(int pos=14;pos>p;--pos)b300_low_state_step(d,h,pos,b300_low_state_get(x,pos));}
    return d>=0?main_rank+Code(d):main_rank-Code(-d);
}'''
s=replace_function(s,'b300_low_cached_drop_rank',low_drop)

# Change only the two production materialization calls. The helper inserted below
# intentionally invokes the low packer and must not be rewritten recursively.
pack_call='b300_pack_low_window_main_mate(m)'
if s.count(pack_call)!=2:raise SystemExit(f'expected two main cache materialization calls, got {s.count(pack_call)}')
s=s.replace(pack_call,'b300_pack_main_transition_cache(m)',2)

high_rank='b300_high_chunk_drop_rank(i,m,p)' if 'b300_high_chunk_drop_rank' in s else 'rank_drop_n_t<TARGET_W>(i,m,p)'
marker='\n\nstatic Code rank_full(MateID m,int width)'
if marker not in s:raise SystemExit('rank_full marker not found')
helpers=r'''

// High forced window keeps only transition positions 14..27. Five base-3
// chunks use 25 bits; signed35 drop delta + 4-bit height fills MateID exactly.
__device__ __forceinline__ bool b300_high_main_state_active(){
    if constexpr(TARGET_W!=28)return false;
    else {constexpr uint32_t HIGH=((uint32_t(1)<<28)-1u)^((uint32_t(1)<<14)-1u);return (D_MAIN_FIXED&HIGH)==0;}
}
__device__ __forceinline__ MateValue b300_high_state_get(MateID x,int p){return MateValue(b300_main_trit_get(x,p-14));}
__device__ __forceinline__ MateValuePair b300_high_state_pair(MateID x,int p){return MateValuePair(uint32_t(b300_high_state_get(x,p-1))|(uint32_t(b300_high_state_get(x,p))<<2));}
__device__ __forceinline__ long long b300_high_state_delta(MateID x){const unsigned long long u=(x>>25)&((MateID(1)<<35)-1ULL);return static_cast<long long>(u<<29)>>29;}
__device__ __forceinline__ int b300_high_state_height(MateID x){return int((x>>60)&15ULL);}
__device__ __forceinline__ MateID b300_high_state_pack_current(MateID x,long long d,int h){constexpr MateID TM=(MateID(1)<<25)-1ULL,DM=(MateID(1)<<35)-1ULL;return (x&TM)|((MateID(d)&DM)<<25)|(MateID(h&15)<<60);}
__device__ __forceinline__ MateID b300_pack_high_main_state(MateID m){MateID trits=0;
#pragma unroll
    for(int c=0;c<5;++c)trits|=MateID(b300_main_trit3(m,14+3*c,28))<<(5*c);return b300_high_state_pack_current(trits,0,1);}
__device__ __forceinline__ MateID b300_pack_main_transition_cache(MateID m){if(b300_high_main_state_active())return b300_pack_high_main_state(m);return b300_pack_low_window_main_mate(m);}
__device__ __forceinline__ void b300_high_state_step(long long&d,int&h,int p,MateValue v){
    if(v==R){d+=static_cast<long long>(D_BLOCK_DP[p-1][h])-static_cast<long long>(D_MAIN_DP[p][h]);--h;}
    else if(v==L){const Code b=D_BLOCK_DP[p-1][h]+(h?D_BLOCK_DP[p-1][h-1]:0),a=D_MAIN_DP[p][h]+(h?D_MAIN_DP[p][h-1]:0);d+=static_cast<long long>(b)-static_cast<long long>(a);++h;}
}
__device__ __forceinline__ MateID b300_high_state_advance(MateID x,int p){long long d=b300_high_state_delta(x);int h=b300_high_state_height(x);b300_high_state_step(d,h,p,b300_high_state_get(x,p));return b300_high_state_pack_current(x,d,h);}
__device__ __forceinline__ Code b300_high_state_drop_rank(Code main_rank,MateID x){const long long d=b300_high_state_delta(x);return d>=0?main_rank+Code(d):main_rank-Code(-d);}
__device__ __forceinline__ Code b300_main_pair_rank_h(Code dst_rank,int p,MateValuePair pair,int h){
    switch(pair){case LR:{const Code d=D_MAIN_DP[p][h]+(h?D_MAIN_DP[p][h-1]:0)+D_MAIN_DP[p-1][h+1];return dst_rank-d;}case NR:{const Code d=D_MAIN_DP[p][h]-D_MAIN_DP[p-1][h];return dst_rank+d;}case NL:{const Code a=D_MAIN_DP[p][h]+(h?D_MAIN_DP[p][h-1]:0),b=D_MAIN_DP[p-1][h]+(h?D_MAIN_DP[p-1][h-1]:0);return dst_rank+(a-b);}default:return dst_rank;}
}
__global__ void b300_init_low_main_state_kernel(MateID*mates,Code n,int p){Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;for(;i<n;i+=stride){MateID x=mates[i];long long d=-static_cast<long long>((x>>25)&0x7fffffffULL);int h=b300_low_state_height(x);for(int pos=14;pos>p;--pos)b300_low_state_step(d,h,pos,b300_low_state_get(x,pos));mates[i]=b300_low_state_pack_current(x,d,h);}}
'''
s=s.replace(marker,helpers+marker,1)

prepare=f'''__device__ __forceinline__ void b300_main_pull_prepare(
    Code i,MateID m,int p,Code nblock,
    Code& pair_j,bool& has_pair,Code& block_j,bool& has_block
){{
    const bool low=b300_low_window_cache_active(),high=b300_high_main_state_active();
    const MateValuePair pair=low?b300_low_state_pair(m,p):(high?b300_high_state_pair(m,p):mpair(m,p));
    const int h=low?b300_low_state_height(m):(high?b300_high_state_height(m):height_before_rank_pos<TARGET_W>(m,p));
    has_pair=(pair==LR||pair==NR||pair==NL);if(has_pair)pair_j=(low||high)?b300_main_pair_rank_h(i,p,pair,h):main_pull_direct_pair_source_rank(i,m,p);
    has_block=false;const MateValue vp=MateValue((uint32_t(pair)>>2)&3u);
    if(nblock&&vp==N){{block_j=low?b300_low_cached_drop_rank(i,m,p):(high?b300_high_state_drop_rank(i,m):{high_rank});has_block=block_j<nblock;}}
}}'''
s=replace_function(s,'b300_main_pull_prepare',prepare)

# ILP2 owns all cached main p>1 updates in the exact forced path.
old='const Count* __restrict__ in,const MateID* __restrict__ mates,Code n,'
if s.count(old)!=1:raise SystemExit(f'ILP2 mate pointer anchor count={s.count(old)}')
s=s.replace(old,'const Count* __restrict__ in,MateID* __restrict__ mates,Code n,',1)
old='uint64_t a0=uint64_t(self0)+pair0+block0;if(a0>=mod)a0-=mod;if(a0>=mod)a0-=mod;out_main[i0]=Count(a0);'
if s.count(old)!=1:raise SystemExit('ILP2 lane0 store anchor not unique')
s=s.replace(old,old+'if(b300_low_window_cache_active())mates[i0]=b300_low_state_advance(m0,p);else if(b300_high_main_state_active())mates[i0]=b300_high_state_advance(m0,p);',1)
old='if(v1){uint64_t a1=uint64_t(self1)+pair1+block1;if(a1>=mod)a1-=mod;if(a1>=mod)a1-=mod;out_main[i1]=Count(a1);}'
if s.count(old)!=1:raise SystemExit('ILP2 lane1 store anchor not unique')
s=s.replace(old,'if(v1){uint64_t a1=uint64_t(self1)+pair1+block1;if(a1>=mod)a1-=mod;if(a1>=mod)a1-=mod;out_main[i1]=Count(a1);if(b300_low_window_cache_active())mates[i1]=b300_low_state_advance(m1,p);else if(b300_high_main_state_active())mates[i1]=b300_high_state_advance(m1,p);}',1)

# Low packed cache starts with base delta/enter height; advance it to the current
# p once after gather. High state naturally starts at p=27 with delta=0,height=1.
old='ck(cudaGetLastError(),"doubleD gather");ck(cudaDeviceSynchronize(),"doubleD gather sync");\n    Count*cur=c.dA,*nxt=c.dB,*dcur=c.dD,*dnext=c.dE;'
new='ck(cudaGetLastError(),"doubleD gather");ck(cudaDeviceSynchronize(),"doubleD gather sync");\n    if(wp.p_hi<15&&useMate&&ms.size){b300_init_low_main_state_kernel<<<bm,threads>>>(c.dMate,ms.size,wp.p_hi);ck(cudaGetLastError(),"init unified low main recurrence");ck(cudaDeviceSynchronize(),"init unified low main recurrence sync");}\n    Count*cur=c.dA,*nxt=c.dB,*dcur=c.dD,*dnext=c.dE;'
if s.count(old)!=1:raise SystemExit(f'gather/init anchor count={s.count(old)}')
s=s.replace(old,new,1)

for required in ('b300_pack_main_transition_cache','b300_low_state_advance(m1,p)','b300_high_state_advance(m1,p)','b300_high_state_drop_rank','b300_low_cached_drop_rank(i,m,p)','b300_init_low_main_state_kernel'):
    if required not in s:raise SystemExit(f'missing unified main recurrence artifact: {required}')
if s.count('b300_pack_main_transition_cache(m)')!=2:raise SystemExit(f'expected two production unified packing calls, got {s.count("b300_pack_main_transition_cache(m)")}')
for stale in ('const MateID* __restrict__ mates','main_pull_direct_pair_source_rank(i,m,p);'):
    if stale in s:raise SystemExit(f'stale unified recurrence artifact: {stale}')
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(s)
print(f'generated {out} from {src}: unified_main_recurrence=1 ilp=2 extra_state_bytes=0 low_trit_bits=25 low_delta_bits=31 high_trit_bits=25 high_delta_bits=35 height_bits=4 mate_hbm_store_per_state_step=8 main_drop_walk_or_table_loads_per_state_step=0 main_height_walk_or_popcount_per_state_step=0')
