#!/usr/bin/env python3
import pathlib,sys
if len(sys.argv)!=3:raise SystemExit('usage: gen-b300-low-block-cache.py INPUT.cu OUTPUT.cu')
src=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);s=src.read_text()
if 'b300_low_cached_drop_rank' not in s or 'D_LOW_DROP_CHUNK' not in s or 'CACHED_BLOCK_MATE' not in s:
    raise SystemExit('low block cache requires block-mate + low-drop-cache + low-drop-chunk transforms')

def once(old,new,label):
    global s
    n=s.count(old)
    if n!=1:raise SystemExit(f'{label}: expected one match got {n}')
    s=s.replace(old,new,1)

once('__constant__ uint32_t* D_LOW_DROP_CHUNK;',
     '__constant__ uint32_t* D_LOW_DROP_CHUNK;\n__constant__ uint32_t* D_LOW14_MATES;\n__constant__ uint32_t D_LOW14_MATE_OFF[MAXW+2];',
     'low14 decode symbols')

# Device helpers are inserted before block_pull_kernel.
marker='template<bool CACHED_BLOCK_MATE>\n__global__ void block_pull_kernel'
if s.count(marker)!=1:raise SystemExit('cached block pull marker not unique')
helper=r'''
__device__ __forceinline__ MateID b300_pack_low_block_cache(Code block_rank,MateID b){
    if(!b300_low_window_cache_active())return b;
    constexpr int LOW=14;
    constexpr uint32_t KMASK=(1u<<13)-1u;
    constexpr uint32_t CODE_MASK=(1u<<(2*13))-1u;
    const uint32_t code=uint32_t((b>>(2*LOW))&CODE_MASK);
    const uint32_t bmask=(D_BLOCK_OCC>>LOW)&KMASK;
    const uint32_t mmask=(D_MAIN_OCC>>(LOW+1))&KMASK;
    const uint32_t begin=D_HIGH_OFFSETS_BLOCK[bmask],end=D_HIGH_OFFSETS_BLOCK[bmask+1];
    uint32_t lo=begin,hi=end;
    while(lo<hi){uint32_t mid=(lo+hi)>>1;if(D_HIGH_ENTRIES_BLOCK[mid].code<code)lo=mid+1;else hi=mid;}
    const HighEntry be=D_HIGH_ENTRIES_BLOCK[lo];
    const uint32_t rel=lo-begin;
    const uint32_t main_base=b300_high_base_lookup(code,mmask,false);
    const uint32_t delta_mag=main_base-be.base;
    const uint32_t local=uint32_t(block_rank-Code(be.base));
    int h=1;
#pragma unroll
    for(int q=12;q>=0;--q){MateValue v=MateValue((code>>(2*q))&3u);if(v==R)--h;else if(v==L)++h;}
    // W28 bounds are build-gated/proved: local<2^18, h<16, rel<2^12,
    // delta_mag<2^30. Exactly 64 bits, no additional per-state scratch.
    return MateID(local)|(MateID(uint32_t(h))<<18)|(MateID(rel)<<22)|(MateID(delta_mag)<<34);
}

struct B300LowBlockDecoded{MateID mate;uint32_t delta_mag;int enter_h;};
__device__ __forceinline__ B300LowBlockDecoded b300_decode_low_block_cache(MateID x){
    const uint32_t local=uint32_t(x&((MateID(1)<<18)-1ULL));
    const int h=int((x>>18)&15ULL);
    const uint32_t rel=uint32_t((x>>22)&((MateID(1)<<12)-1ULL));
    const uint32_t delta_mag=uint32_t(x>>34);
    const uint32_t bmask=(D_BLOCK_OCC>>14)&((1u<<13)-1u);
    const HighEntry e=D_HIGH_ENTRIES_BLOCK[D_HIGH_OFFSETS_BLOCK[bmask]+rel];
    const uint32_t low=__ldg(D_LOW14_MATES+D_LOW14_MATE_OFF[h]+local);
    return{MateID(low)|(MateID(e.code)<<28),delta_mag,h};
}

__device__ __forceinline__ long long b300_low_free_drop_delta(MateID low15,int enter_h,int p){
    constexpr int HC=MAXW+2;
    long long delta=0;int h=enter_h;
    const int n=14-p,full=n>>2,rem=n&3;
#pragma unroll
    for(int c=0;c<3;++c){
        if(c>=full)continue;
        const int lo=11-4*c;
        const uint32_t code=uint32_t((low15>>(2*lo))&0xffULL);
        const uint32_t z=__ldg(D_LOW_DROP_CHUNK+(size_t(c)*HC+size_t(h))*256u+code);
        delta+=static_cast<long long>(int32_t(z<<8)>>8);h=int(z>>24);
    }
    int pos=14-(full<<2);
#pragma unroll
    for(int r=0;r<3;++r,--pos){
        if(r>=rem)continue;
        const MateValue v=mget(low15,pos);
        if(v==R){
            delta+=static_cast<long long>(D_FULL_DP[pos-1][h])-static_cast<long long>(D_FULL_DP[pos][h]);--h;
        }else if(v==L){
            const Code bm=D_FULL_DP[pos-1][h]+(h?D_FULL_DP[pos-1][h-1]:0);
            const Code am=D_FULL_DP[pos][h]+(h?D_FULL_DP[pos][h-1]:0);
            delta+=static_cast<long long>(bm)-static_cast<long long>(am);++h;
        }
    }
    return delta;
}

__device__ __forceinline__ Code b300_low_block_lift_rank(
    Code block_rank,MateID block_mate,uint32_t delta_mag,int enter_h,int insert_pos
){
    const MateID full=minsert(block_mate,insert_pos,N);
    constexpr MateID LOW15_MASK=(MateID(1)<<30)-1ULL;
    const long long local=b300_low_free_drop_delta(full&LOW15_MASK,enter_h,insert_pos);
    // block_rank = main_rank - delta_mag + local
    const long long d=static_cast<long long>(delta_mag)-local;
    return d>=0?block_rank+Code(d):block_rank-Code(-d);
}

'''
s=s.replace(marker,helper+marker,1)

# Pack at the one-time group/window block-mate materialization. Both paths have
# the group-local block rank i available.
once('if(mates)mates[i]=m;','if(mates)mates[i]=b300_pack_low_block_cache(i,m);','packed block gather')
once('for(;i<n;i+=stride)mates[i]=unrank_group_t<TARGET_W-1>(i,D_BLOCK_FIXED,D_BLOCK_OCC,D_BLOCK_DP);',
     'for(;i<n;i+=stride){MateID m=unrank_group_t<TARGET_W-1>(i,D_BLOCK_FIXED,D_BLOCK_OCC,D_BLOCK_DP);mates[i]=b300_pack_low_block_cache(i,m);}',
     'packed interval block materialize')

# Decode once per destination. High-window caches remain literal MateID.
old='MateID b;if constexpr(CACHED_BLOCK_MATE)b=block_mates[i];else b=unrank_group_t<TARGET_W-1>(i,D_BLOCK_FIXED,D_BLOCK_OCC,D_BLOCK_DP);'
new='MateID cached=0,b;uint32_t low_delta_mag=0;int low_enter_h=0;bool low_packed=false;if constexpr(CACHED_BLOCK_MATE){cached=block_mates[i];low_packed=b300_low_window_cache_active();if(low_packed){auto z=b300_decode_low_block_cache(cached);b=z.mate;low_delta_mag=z.delta_mag;low_enter_h=z.enter_h;}else b=cached;}else b=unrank_group_t<TARGET_W-1>(i,D_BLOCK_FIXED,D_BLOCK_OCC,D_BLOCK_DP);'
once(old,new,'decode packed block cache')

# Generic path, or an already-installed high-chunk path, both retain a low
# rank_lift fallback. Replace only that fallback with the packed O(1)+chunk lift.
plain='Code j=rank_lift_n_t<TARGET_W>(i,b,p);'
chunked='Code j=p>=15?b300_high_chunk_lift_rank(i,minsert(b,p,N),p):rank_lift_n_t<TARGET_W>(i,b,p);'
if plain in s:
    once(plain,'Code j=low_packed?b300_low_block_lift_rank(i,b,low_delta_mag,low_enter_h,p):rank_lift_n_t<TARGET_W>(i,b,p);','endpoint low packed lift')
elif chunked in s:
    once(chunked,'Code j=low_packed?b300_low_block_lift_rank(i,b,low_delta_mag,low_enter_h,p):(p>=15?b300_high_chunk_lift_rank(i,minsert(b,p,N),p):rank_lift_n_t<TARGET_W>(i,b,p));','endpoint low packed/high chunk lift')
else:raise SystemExit('endpoint lift anchor not found')

plain='Code base_rank=rank_lift_n_t<TARGET_W>(i,b,p-1);'
chunked='Code base_rank=p>=15?b300_high_chunk_lift_rank(i,d,p-1):rank_lift_n_t<TARGET_W>(i,b,p-1);'
if plain in s:
    once(plain,'Code base_rank=low_packed?b300_low_block_lift_rank(i,b,low_delta_mag,low_enter_h,p-1):rank_lift_n_t<TARGET_W>(i,b,p-1);','closure low packed lift')
elif chunked in s:
    once(chunked,'Code base_rank=low_packed?b300_low_block_lift_rank(i,b,low_delta_mag,low_enter_h,p-1):(p>=15?b300_high_chunk_lift_rank(i,d,p-1):rank_lift_n_t<TARGET_W>(i,b,p-1));','closure low packed/high chunk lift')
else:raise SystemExit('closure lift anchor not found')

# Build the unrestricted 14-cell suffix decode table once. The per-height local
# rank is already stored in the 18-bit cache field.
main_marker='\nint main(int argc,char**argv){'
if s.count(main_marker)!=1:raise SystemExit('main marker not unique')
host=r'''

static std::vector<uint32_t> build_low14_mate_decode(std::array<uint32_t,MAXW+2>& off){
    std::vector<uint32_t> out;
    for(int h=0;h<MAXW+2;++h){
        off[h]=uint32_t(out.size());
        const Code cnt=H_DP[14][h];
        for(Code r=0;r<cnt;++r){
            MateID m=unrank_suffix_host(r,14,h,0,0,H_DP);
            out.push_back(uint32_t(m));
        }
    }
    return out;
}
'''
s=s.replace(main_marker,host+main_marker,1)

alloc_marker='\n\n    Code mainN=H_DP[W][1],blockN=H_DP[W-1][1];'
if s.count(alloc_marker)!=1:raise SystemExit('auth allocation marker not unique')
alloc=r'''

    std::array<uint32_t,MAXW+2> low14Off{};
    std::vector<uint32_t> low14MateHost=build_low14_mate_decode(low14Off);
    uint32_t* low14MateD[MAXGPU]{};
    for(int d=0;d<ng;++d){
        ck(cudaSetDevice(d),"set low14 mate decode device");
        ck(cudaMalloc(&low14MateD[d],low14MateHost.size()*sizeof(uint32_t)),"low14 mate decode alloc");
        ck(cudaMemcpy(low14MateD[d],low14MateHost.data(),low14MateHost.size()*sizeof(uint32_t),cudaMemcpyHostToDevice),"low14 mate decode copy");
        ck(cudaMemcpyToSymbol(D_LOW14_MATES,&low14MateD[d],sizeof(low14MateD[d])),"low14 mate decode ptr");
        ck(cudaMemcpyToSymbol(D_LOW14_MATE_OFF,low14Off.data(),sizeof(low14Off)),"low14 mate decode offsets");
    }
    std::cerr<<"low_block_cache low14_decode_entries="<<low14MateHost.size()<<" bytes="<<low14MateHost.size()*sizeof(uint32_t)<<" packed_bits=64 extra_per_state_bytes=0\n";'''
s=s.replace(alloc_marker,alloc+alloc_marker,1)

out.parent.mkdir(parents=True,exist_ok=True);out.write_text(s)
print(f'generated {out} from {src}: low_block_cache=1 packed_bits=64 extra_per_state_bytes=0 low_rank_bits=18 height_bits=4 high_index_bits=12 base_delta_bits=30 low_lift_prefix_walk=0')
