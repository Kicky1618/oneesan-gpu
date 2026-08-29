#!/usr/bin/env python3
import pathlib, sys

if len(sys.argv) != 3:
    raise SystemExit('usage: gen-b300-low-window-drop-cache.py INPUT.cu OUTPUT.cu')
src=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2]); s=src.read_text()
if 'main_pull_direct_pair_source_rank' not in s or 'materialize_main_mates_kernel' not in s:
    raise SystemExit('low-window drop cache requires main-mate + main-pull transforms first')

def once(old,new,label):
    global s
    n=s.count(old)
    if n!=1: raise SystemExit(f'{label}: expected one match got {n}')
    s=s.replace(old,new,1)

# W28/K13 low window: high positions 15..27 are fixed, low positions 0..14
# are free.  Cache only what the pull hot path needs:
#   bits  0..29 : low 15 MateValue symbols
#   bits 30..59 : main-high-base - block-high-base (proved < 2^30)
#   bits 60..63 : height entering low suffix (proved <= 15)
# The cache remains exactly one MateID (8 bytes) per main state.
marker='__global__ void gather_main_kernel('
if s.count(marker)!=1: raise SystemExit('gather_main_kernel marker not unique')
helper=r'''
__device__ __forceinline__ bool b300_low_window_cache_active(){
    if constexpr(TARGET_W!=28 || HIGH_LUT_K!=13)return false;
    else {
        constexpr uint32_t HM=((1u<<13)-1u)<<15;
        constexpr uint32_t WM=(1u<<28)-1u;
        return (D_MAIN_FIXED&HM)==HM && (D_MAIN_FIXED&(WM^HM))==0;
    }
}

__device__ __forceinline__ uint32_t b300_high_base_lookup(
    uint32_t code,uint32_t mask,bool block
){
    HighEntry* entries=block?D_HIGH_ENTRIES_BLOCK:D_HIGH_ENTRIES_MAIN;
    uint32_t* offsets=block?D_HIGH_OFFSETS_BLOCK:D_HIGH_OFFSETS_MAIN;
    uint32_t lo=offsets[mask],hi=offsets[mask+1];
    while(lo<hi){uint32_t mid=(lo+hi)>>1;if(entries[mid].code<code)lo=mid+1;else hi=mid;}
    return entries[lo].base;
}

__device__ __forceinline__ MateID b300_pack_low_window_main_mate(MateID m){
    if(!b300_low_window_cache_active())return m;
    constexpr int LOW=15;
    constexpr MateID LOW_MASK=(MateID(1)<<(2*LOW))-1ULL;
    constexpr uint32_t CODE_MASK=(1u<<(2*13))-1u;
    const uint32_t code=uint32_t((m>>(2*LOW))&CODE_MASK);
    const uint32_t mmask=(D_MAIN_OCC>>LOW)&((1u<<13)-1u);
    const uint32_t bmask=(D_BLOCK_OCC>>(LOW-1))&((1u<<13)-1u);
    const uint32_t main_base=b300_high_base_lookup(code,mmask,false);
    const uint32_t block_base=b300_high_base_lookup(code,bmask,true);
    const uint32_t delta_mag=main_base-block_base;
    const MateID high=m>>(2*LOW);
    constexpr MateID EVEN=0x5555555555555555ULL;
    const int nr=__popcll(high&EVEN),nl=__popcll((high>>1)&EVEN);
    const uint32_t enter_h=uint32_t(1+nl-nr);
    return (m&LOW_MASK)|(MateID(delta_mag)<<30)|(MateID(enter_h)<<60);
}

__device__ __forceinline__ int b300_low_cached_height_before(MateID cached,int p){
    constexpr MateID LOW_MASK=(MateID(1)<<30)-1ULL;
    constexpr MateID EVEN=0x5555555555555555ULL;
    const MateID low=cached&LOW_MASK;
    const MateID lower=(MateID(1)<<(2*(p+1)))-1ULL;
    const MateID pm=EVEN&LOW_MASK&~lower;
    const int nr=__popcll(low&pm),nl=__popcll((low>>1)&pm);
    return int((cached>>60)&15ULL)+nl-nr;
}

__device__ __forceinline__ Code b300_low_cached_drop_rank(Code main_rank,MateID cached,int p){
    constexpr MateID LOW_MASK=(MateID(1)<<30)-1ULL;
    const MateID low=cached&LOW_MASK;
    long long delta=-long long((cached>>30)&((MateID(1)<<30)-1ULL));
    int h=int((cached>>60)&15ULL);
#pragma unroll
    for(int pos=14;pos>=0;--pos){
        if(pos<=p)continue;
        const MateValue v=mget(low,pos);
        if(v==R){
            delta+=long long(D_FULL_DP[pos-1][h])-long long(D_FULL_DP[pos][h]);
            --h;
        }else if(v==L){
            const Code bm=D_FULL_DP[pos-1][h]+(h?D_FULL_DP[pos-1][h-1]:0);
            const Code am=D_FULL_DP[pos][h]+(h?D_FULL_DP[pos][h-1]:0);
            delta+=long long(bm)-long long(am);
            ++h;
        }
    }
    return delta>=0?main_rank+Code(delta):main_rank-Code(-delta);
}

'''
s=s.replace(marker,helper+marker,1)

# Fuse packing into both non-interval gather and interval materialization.
once('if(mates)mates[i]=m;','if(mates)mates[i]=b300_pack_low_window_main_mate(m);','gather packed mate')
once('for(;i<n;i+=stride)mates[i]=unrank_group_t<TARGET_W>(i,D_MAIN_FIXED,D_MAIN_OCC,D_MAIN_DP);',
     'for(;i<n;i+=stride){MateID m=unrank_group_t<TARGET_W>(i,D_MAIN_FIXED,D_MAIN_OCC,D_MAIN_DP);mates[i]=b300_pack_low_window_main_mate(m);}',
     'interval packed mate')

# Direct pair-rank still uses the cached low symbols, but its entering height
# must come from the packed four-bit field rather than the omitted high MateID.
once('const int h=height_before_rank_pos<TARGET_W>(m,p);',
     'const int h=b300_low_window_cache_active()?b300_low_cached_height_before(m,p):height_before_rank_pos<TARGET_W>(m,p);',
     'direct pair cached height')

# Main->blocked inverse rank: low window uses the packed high-base delta and a
# short free-suffix recurrence; high window retains the proven generic delta.
once('if(nblock&&mget(m,p)==N){Code j=rank_drop_n_t<TARGET_W>(i,m,p);if(j<nblock)acc+=in_block[j];}',
     'if(nblock&&mget(m,p)==N){Code j=b300_low_window_cache_active()?b300_low_cached_drop_rank(i,m,p):rank_drop_n_t<TARGET_W>(i,m,p);if(j<nblock)acc+=in_block[j];}',
     'cached low drop rank')

# p=1 uses the legacy source-centric kernel.  The low-window cache intentionally
# omits the high labels, so unrank the full MateID for this single boundary step.
once('if(ms.size){if(useMate)main_group_kernel<true><<<bm,threads,0,c.sMain>>>(cur,c.dMate,ms.size,nxt,dnext,p);else main_group_kernel<false><<<bm,threads,0,c.sMain>>>(cur,nullptr,ms.size,nxt,dnext,p);}',
     'if(ms.size)main_group_kernel<false><<<bm,threads,0,c.sMain>>>(cur,nullptr,ms.size,nxt,dnext,p);',
     'p1 uncached full mate')

out.parent.mkdir(parents=True,exist_ok=True);out.write_text(s)
print(f'generated {out} from {src}: low_window_packed_main_cache=1 cache_bytes_per_main=8 high_prefix_walk=0 low_drop_max_steps=12 p1_uncached=1 w28_k13_only=1')
