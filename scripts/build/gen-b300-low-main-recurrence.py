#!/usr/bin/env python3
from __future__ import annotations
import pathlib,sys

if len(sys.argv)!=3:raise SystemExit('usage: gen-b300-low-main-recurrence.py INPUT.cu OUTPUT.cu')
src=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);s=src.read_text()
for req in ('b300_low_window_cache_active','b300_low_cached_drop_rank','main_pull_direct_pair_source_rank','dMate'):
    if req not in s:raise SystemExit(f'low-main recurrence requires artifact: {req}')

def replace_function(text:str,name:str,new:str)->str:
    p=text.find(name+'(')
    if p<0:raise SystemExit(f'function not found: {name}')
    start=text.rfind('\n',0,p)+1
    brace=text.find('{',p)
    if brace<0:raise SystemExit(f'function brace not found: {name}')
    depth=0;end=-1
    for i in range(brace,len(text)):
        if text[i]=='{':depth+=1
        elif text[i]=='}':
            depth-=1
            if depth==0:end=i+1;break
    if end<0:raise SystemExit(f'function end not found: {name}')
    return text[:start]+new+text[end:]

pack='''__device__ __forceinline__ uint32_t b300_low_trit_chunk(MateID m,int base){
    return uint32_t(mget(m,base))+3u*uint32_t(mget(m,base+1))+9u*uint32_t(mget(m,base+2));
}
__device__ __forceinline__ MateValue b300_low_state_get(MateID x,int p){
    if(!b300_low_window_cache_active())return mget(x,p);
    const uint32_t z=uint32_t((x>>(5*(p/3)))&31ULL);const int r=p%3;
    uint32_t v;if(r==0){uint32_t q=z/3;v=z-q*3;}else if(r==1){uint32_t q=z/3;v=q-(q/3)*3;}else v=z/9;
    return MateValue(v);
}
__device__ __forceinline__ MateValuePair b300_low_state_pair(MateID x,int p){
    if(!b300_low_window_cache_active())return mpair(x,p);
    return MateValuePair(uint32_t(b300_low_state_get(x,p-1))|(uint32_t(b300_low_state_get(x,p))<<2));
}
__device__ __forceinline__ long long b300_low_state_delta(MateID x){
    const unsigned long long u=(x>>25)&0x7fffffffULL;return static_cast<long long>(u<<33)>>33;
}
__device__ __forceinline__ int b300_low_state_height(MateID x){return int((x>>56)&15ULL);}
__device__ __forceinline__ bool b300_low_state_ready(MateID x){return ((x>>60)&1ULL)!=0;}
__device__ __forceinline__ MateID b300_low_state_pack_current(MateID x,long long d,int h){
    constexpr MateID TM=(MateID(1)<<25)-1ULL,DM=(MateID(1)<<31)-1ULL;
    return (x&TM)|((MateID(d)&DM)<<25)|(MateID(h&15)<<56)|(MateID(1)<<60);
}
__device__ __forceinline__ void b300_low_state_step(long long&d,int&h,int p,MateValue v){
    if(v==R){d+=static_cast<long long>(D_FULL_DP[p-1][h])-static_cast<long long>(D_FULL_DP[p][h]);--h;}
    else if(v==L){const Code b=D_FULL_DP[p-1][h]+(h?D_FULL_DP[p-1][h-1]:0),a=D_FULL_DP[p][h]+(h?D_FULL_DP[p][h-1]:0);d+=static_cast<long long>(b)-static_cast<long long>(a);++h;}
}
__device__ __forceinline__ MateID b300_low_state_advance(MateID x,int p){
    if(!b300_low_window_cache_active()||!b300_low_state_ready(x))return x;
    long long d=b300_low_state_delta(x);int h=b300_low_state_height(x);b300_low_state_step(d,h,p,b300_low_state_get(x,p));return b300_low_state_pack_current(x,d,h);
}
__device__ __forceinline__ MateID b300_pack_low_window_main_mate(MateID m){
    if(!b300_low_window_cache_active())return m;
    MateID trits=0;
#pragma unroll
    for(int c=0;c<5;++c)trits|=MateID(b300_low_trit_chunk(m,3*c))<<(5*c);
    constexpr int LOW=15;constexpr uint32_t CODE_MASK=(1u<<(2*13))-1u;
    const uint32_t code=uint32_t((m>>(2*LOW))&CODE_MASK),mmask=(D_MAIN_OCC>>LOW)&((1u<<13)-1u),bmask=(D_BLOCK_OCC>>(LOW-1))&((1u<<13)-1u);
    const uint32_t main_base=b300_high_base_lookup(code,mmask,false),block_base=b300_high_base_lookup(code,bmask,true),delta_mag=main_base-block_base;
    const MateID high=m>>(2*LOW);constexpr MateID EVEN=0x5555555555555555ULL;const int nr=__popcll(high&EVEN),nl=__popcll((high>>1)&EVEN);const uint32_t enter_h=uint32_t(1+nl-nr);
    return trits|(MateID(delta_mag)<<25)|(MateID(enter_h)<<56);
}'''
s=replace_function(s,'b300_pack_low_window_main_mate',pack)

height='''__device__ __forceinline__ int b300_low_cached_height_before(MateID x,int p){
    if(b300_low_state_ready(x))return b300_low_state_height(x);
    int h=b300_low_state_height(x);
    for(int pos=14;pos>p;--pos){MateValue v=b300_low_state_get(x,pos);h+=(v==L)-(v==R);}return h;
}'''
s=replace_function(s,'b300_low_cached_height_before',height)

drop='''__device__ __forceinline__ Code b300_low_cached_drop_rank(Code main_rank,MateID x,int p){
    long long d;
    if(b300_low_state_ready(x))d=b300_low_state_delta(x);
    else {d=-static_cast<long long>((x>>25)&0x7fffffffULL);int h=b300_low_state_height(x);for(int pos=14;pos>p;--pos)b300_low_state_step(d,h,pos,b300_low_state_get(x,pos));}
    return d>=0?main_rank+Code(d):main_rank-Code(-d);
}'''
s=replace_function(s,'b300_low_cached_drop_rank',drop)

marker='\n\nstatic Code rank_full(MateID m,int width)'
init='''
__global__ void b300_init_low_main_state_kernel(MateID*mates,Code n,int p){
    Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;
    for(;i<n;i+=stride){MateID x=mates[i];long long d=-static_cast<long long>((x>>25)&0x7fffffffULL);int h=b300_low_state_height(x);for(int pos=14;pos>p;--pos)b300_low_state_step(d,h,pos,b300_low_state_get(x,pos));mates[i]=b300_low_state_pack_current(x,d,h);}
}
'''
if marker not in s:raise SystemExit('rank_full marker not found')
s=s.replace(marker,init+marker,1)

# Pair rank keeps the pair already decoded by the caller and consumes the cached
# current height. High-window states keep the original MateID representation.
s=s.replace('Code main_pull_direct_pair_source_rank(Code dst_rank,MateID m,int p){','Code main_pull_direct_pair_source_rank(Code dst_rank,MateID m,int p,MateValuePair pair){',1)
old='const int h=b300_low_window_cache_active()?b300_low_cached_height_before(m,p):height_before_rank_pos<TARGET_W>(m,p);'
if s.count(old)!=1:raise SystemExit(f'pair height anchor count={s.count(old)}')
s=s.replace(old,'const int h=b300_low_window_cache_active()?b300_low_state_height(m):height_before_rank_pos<TARGET_W>(m,p);',1)
if s.count('switch(mpair(m,p)){')<1:raise SystemExit('pair switch anchor missing')
s=s.replace('switch(mpair(m,p)){','switch(pair){',1)

# Decode once per destination and reuse the high symbol for the blocked-source gate.
s=s.replace('const MateValuePair pair=mpair(m,p);','const MateValuePair pair=b300_low_state_pair(m,p);')
s=s.replace('main_pull_direct_pair_source_rank(i,m,p);','main_pull_direct_pair_source_rank(i,m,p,pair);')
s=s.replace('if(nblock&&mget(m,p)==N){','if(nblock&&MateValue((uint32_t(pair)>>2)&3u)==N){')

# Cached main mate must be writable so each p advances delta+height in-place.
s=s.replace('__global__ void main_pull_kernel(const Count*in,const MateID*mates,','__global__ void main_pull_kernel(const Count*in,MateID*mates,',1)
s=s.replace('const Count* __restrict__ in,const MateID* __restrict__ mates,Code n,','const Count* __restrict__ in,MateID* __restrict__ mates,Code n,')

# Generic kernel store.
old='out_main[i]=Count(acc);'
if old in s:s=s.replace(old,'out_main[i]=Count(acc);if constexpr(CACHED_MATE){if(b300_low_window_cache_active())mates[i]=b300_low_state_advance(m,p);}',1)
# ILP stores, whichever experimental kernel is present.
for lane in range(4):
    old=f'out_main[i{lane}]=Count(a{lane});'
    if old in s:s=s.replace(old,old+f'if(b300_low_window_cache_active())mates[i{lane}]=b300_low_state_advance(m{lane},p);')

# Initialize recurrent state once after structural mate materialization, before p loop.
old='ck(cudaGetLastError(),"doubleD gather");ck(cudaDeviceSynchronize(),"doubleD gather sync");\n    Count*cur=c.dA,*nxt=c.dB,*dcur=c.dD,*dnext=c.dE;'
new='ck(cudaGetLastError(),"doubleD gather");ck(cudaDeviceSynchronize(),"doubleD gather sync");\n    if(wp.p_hi<15&&useMate&&ms.size){b300_init_low_main_state_kernel<<<bm,threads>>>(c.dMate,ms.size,wp.p_hi);ck(cudaGetLastError(),"init low main recurrence");ck(cudaDeviceSynchronize(),"init low main recurrence sync");}\n    Count*cur=c.dA,*nxt=c.dB,*dcur=c.dD,*dnext=c.dE;'
if s.count(old)!=1:raise SystemExit(f'gather/init anchor count={s.count(old)}')
s=s.replace(old,new,1)

for required in ('b300_init_low_main_state_kernel','b300_low_state_advance','b300_low_state_pair','MateID* __restrict__ mates','main_pull_direct_pair_source_rank(i,m,p,pair)'):
    if required not in s:raise SystemExit(f'missing low-main recurrence artifact: {required}')
for forbidden in ('b300_low_cached_height_before(m,p):height_before_rank_pos','main_pull_direct_pair_source_rank(i,m,p);'):
    if forbidden in s:raise SystemExit(f'stale low-main artifact: {forbidden}')
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(s)
print(f'generated {out} from {src}: low_main_recurrence=1 extra_state_bytes=0 trit_bits=25 signed_delta_bits=31 height_bits=4 mate_hbm_store_per_state_step=8 low_drop_table_loads_per_state_step=0 low_height_popcounts_per_state_step=0')
