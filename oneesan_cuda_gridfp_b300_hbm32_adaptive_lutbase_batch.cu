#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

using Count = uint32_t;
using MateID = unsigned long long;
using Code = unsigned long long;
static constexpr int MAXW=28, MAXGPU=8;
#ifndef IO_TILE_ELEMS
#define IO_TILE_ELEMS 65536ULL
#endif
#ifndef TARGET_W
#define TARGET_W 28
#endif
#ifndef LOW_LUT_K
#define LOW_LUT_K 0
#endif
#ifndef HIGH_LUT_K
#define HIGH_LUT_K 0
#endif
#ifndef BASE_LUT_K
#define BASE_LUT_K LOW_LUT_K
#endif

enum MateValue:uint8_t{N=0,R=1,L=2,X=3};
enum MateValuePair:uint8_t{NN=0x0,NR=0x1,NL=0x2,NX=0x3,RN=0x4,RR=0x5,RL=0x6,RX=0x7,LN=0x8,LR=0x9,LL=0xa,LX=0xb,XN=0xc,XR=0xd,XL=0xe,XX=0xf};

static Code H_DP[MAXW+1][MAXW+2];
__constant__ Code D_FULL_DP[MAXW+1][MAXW+2];
__constant__ Code D_MAIN_DP[MAXW+1][MAXW+2];
__constant__ Code D_BLOCK_DP[MAXW+1][MAXW+2];
__constant__ uint32_t D_MAIN_FIXED,D_MAIN_OCC,D_BLOCK_FIXED,D_BLOCK_OCC;
__constant__ int D_MAIN_W,D_BLOCK_W,D_NGPU;
__constant__ Count D_MOD;
__constant__ Count* D_MAIN_PTR[MAXGPU];
__constant__ Count* D_BLOCK_PTR[MAXGPU];
__constant__ Code D_MAIN_CHUNK,D_BLOCK_CHUNK;
struct LowEntry{uint32_t mate,full_rank;};
__constant__ LowEntry* D_LOW_ENTRIES;
__constant__ uint32_t* D_LOW_OFFSETS;
__constant__ uint32_t* D_LOW_LOCAL_RANK;
struct HighEntry{uint32_t code,base;};
__constant__ HighEntry* D_HIGH_ENTRIES_MAIN;
__constant__ HighEntry* D_HIGH_ENTRIES_BLOCK;
__constant__ uint32_t* D_HIGH_OFFSETS_MAIN;
__constant__ uint32_t* D_HIGH_OFFSETS_BLOCK;

__host__ __device__ static inline MateValue mget(MateID m,int k){return MateValue((m>>(2*k))&3ULL);}
__host__ __device__ static inline MateValuePair mpair(MateID m,int p){return MateValuePair((m>>(2*(p-1)))&15ULL);}
__host__ __device__ static inline MateID mset(MateID m,int k,MateValue v){MateID z=3ULL<<(2*k);return (m&~z)|(MateID(v)<<(2*k));}
__host__ __device__ static inline MateID msetpair(MateID m,int p,MateValuePair v){MateID z=15ULL<<(2*(p-1));return (m&~z)|(MateID(v)<<(2*(p-1)));}
__host__ __device__ static inline MateID mshrink(MateID m,int k){MateID mask=(1ULL<<(2*k))-1ULL;return((m&~mask)>>2)|(m&mask);}
__host__ __device__ static inline MateID minsert(MateID m,int k,MateValue v){MateID lowmask=k?((1ULL<<(2*k))-1ULL):0ULL;MateID lo=m&lowmask,hi=m&~lowmask;return lo|(MateID(v)<<(2*k))|(hi<<2);}

static void ck(cudaError_t e,const char* w){if(e!=cudaSuccess){std::cerr<<w<<": "<<cudaGetErrorString(e)<<"\n";std::exit(1);}}
static void build_full_dp(){for(int h=0;h<=MAXW+1;++h)H_DP[0][h]=(h==0);for(int w=1;w<=MAXW;++w)for(int h=0;h<=MAXW;++h){Code x=H_DP[w-1][h];if(h>0)x+=H_DP[w-1][h-1];if(h<MAXW+1)x+=H_DP[w-1][h+1];H_DP[w][h]=x;}}

struct GroupSpec{int width=0;uint32_t fixed=0,occ=0;Code dp[MAXW+1][MAXW+2]{};Code size=0;};
static GroupSpec make_spec(int width,uint32_t fixed,uint32_t occ){GroupSpec s;s.width=width;s.fixed=fixed;s.occ=occ;for(int h=0;h<=MAXW+1;++h)s.dp[0][h]=(h==0);for(int w=1;w<=width;++w){int pos=w-1;bool f=(fixed>>pos)&1u,o=(occ>>pos)&1u;for(int h=0;h<=MAXW;++h){Code x=0;if(!f||!o)x+=s.dp[w-1][h];if(!f||o){if(h>0)x+=s.dp[w-1][h-1];if(h<MAXW+1)x+=s.dp[w-1][h+1];}s.dp[w][h]=x;}}s.size=s.dp[width][1];return s;}

static bool allowed_host(uint32_t fixed,uint32_t occ,int pos,MateValue v){if(!((fixed>>pos)&1u))return v!=X;bool o=(occ>>pos)&1u;return o?(v==R||v==L):(v==N);}
static MateID unrank_suffix_host(Code rank,int width,int start_h,uint32_t fixed,uint32_t occ,const Code dp[MAXW+1][MAXW+2]){
    MateID m=0;int h=start_h;
    for(int pos=width-1;pos>=0;--pos){
        if(allowed_host(fixed,occ,pos,N)){Code z=dp[pos][h];if(rank<z)continue;rank-=z;}
        if(h>0&&allowed_host(fixed,occ,pos,R)){Code z=dp[pos][h-1];if(rank<z){m|=MateID(R)<<(2*pos);--h;continue;}rank-=z;}
        m|=MateID(L)<<(2*pos);++h;
    }
    return m;
}
static Code rank_full_suffix_host(MateID m,int width,int start_h){Code rank=0;int h=start_h;for(int pos=width-1;pos>=0;--pos){auto v=mget(m,pos);if(v>N)rank+=H_DP[pos][h];if(v>R&&h>0)rank+=H_DP[pos][h-1];if(v==R)--h;else if(v==L)++h;}return rank;}
struct LowTablesHost{std::vector<LowEntry> entries;std::vector<uint32_t> offsets,local_rank;};
static LowTablesHost build_low_tables(){
    LowTablesHost t;
    if constexpr(LOW_LUT_K==0)return t;
    constexpr uint32_t NM=1u<<LOW_LUT_K;constexpr uint32_t NC=1u<<(2*LOW_LUT_K);
    t.offsets.resize(size_t(NM)*(LOW_LUT_K+2));t.local_rank.assign(NC,0xffffffffu);
    constexpr uint32_t FIX=NM-1;
    for(uint32_t mask=0;mask<NM;++mask){
        auto sp=make_spec(LOW_LUT_K,FIX,mask);
        for(int h=0;h<LOW_LUT_K+2;++h){
            t.offsets[size_t(mask)*(LOW_LUT_K+2)+h]=(uint32_t)t.entries.size();
            Code cnt=sp.dp[LOW_LUT_K][h];
            for(Code r=0;r<cnt;++r){MateID m=unrank_suffix_host(r,LOW_LUT_K,h,FIX,mask,sp.dp);Code fr=rank_full_suffix_host(m,LOW_LUT_K,h);if(fr>0xffffffffULL){std::cerr<<"low full rank overflow\n";std::exit(6);}uint32_t code=(uint32_t)m;t.entries.push_back({code,(uint32_t)fr});t.local_rank[code]=(uint32_t)r;}
        }
    }
    std::cerr<<"low_lut K="<<LOW_LUT_K<<" entries="<<t.entries.size()<<" entries_mib="<<double(t.entries.size()*sizeof(LowEntry))/(1<<20)<<" dense_rank_mib="<<double(t.local_rank.size()*sizeof(uint32_t))/(1<<20)<<"\n";
    return t;
}

struct Interval{Code global,local,len;};
struct HighTablesHost{std::vector<HighEntry> entries;std::vector<uint32_t> offsets;};
static void high_enum_rec(int pos,int h,uint32_t occ,uint32_t code,int lowlen,uint64_t&base,std::vector<HighEntry>&entries){
    if(pos<0){Code cnt=H_DP[lowlen][h];if(!cnt)return;if(base>0xffffffffULL||base+cnt>0x100000000ULL){std::cerr<<"high base overflow\n";std::exit(7);}entries.push_back({code,(uint32_t)base});base+=cnt;return;}
    bool o=(occ>>pos)&1u;
    if(!o){high_enum_rec(pos-1,h,occ,code,lowlen,base,entries);return;}
    if(h>0)high_enum_rec(pos-1,h-1,occ,code|(uint32_t(R)<<(2*pos)),lowlen,base,entries);
    high_enum_rec(pos-1,h+1,occ,code|(uint32_t(L)<<(2*pos)),lowlen,base,entries);
}
static HighTablesHost build_high_tables(int width){
    HighTablesHost t;if constexpr(HIGH_LUT_K==0)return t;
    constexpr uint32_t NM=1u<<HIGH_LUT_K;int lowlen=width-HIGH_LUT_K;
    t.offsets.resize(NM+1);
    for(uint32_t mask=0;mask<NM;++mask){t.offsets[mask]=(uint32_t)t.entries.size();uint64_t base=0;high_enum_rec(HIGH_LUT_K-1,1,mask,0,lowlen,base,t.entries);}t.offsets[NM]=(uint32_t)t.entries.size();
    std::cerr<<"high_lut width="<<width<<" K="<<HIGH_LUT_K<<" entries="<<t.entries.size()<<" entries_mib="<<double(t.entries.size()*sizeof(HighEntry))/(1<<20)<<" offsets_mib="<<double(t.offsets.size()*sizeof(uint32_t))/(1<<20)<<"\n";return t;
}

static void add_interval(std::vector<Interval>&out,Code g,Code l,Code n){if(!n)return;if(!out.empty()&&out.back().global+out.back().len==g&&out.back().local+out.back().len==l)out.back().len+=n;else out.push_back({g,l,n});}
static void intervals_rec(const GroupSpec&s,int pos,int h,Code gbase,Code lbase,std::vector<Interval>&out){
    if(pos<0){if(h==0)add_interval(out,gbase,lbase,1);return;}
    uint32_t lower=(pos==31)?0xffffffffu:((1u<<(pos+1))-1u);
    if((s.fixed&lower)==0){add_interval(out,gbase,lbase,H_DP[pos+1][h]);return;}
    bool f=(s.fixed>>pos)&1u,o=(s.occ>>pos)&1u;
    Code gsz=H_DP[pos][h];
    if(!f||!o){Code lsz=s.dp[pos][h];intervals_rec(s,pos-1,h,gbase,lbase,out);lbase+=lsz;}gbase+=gsz;
    if(h>0){gsz=H_DP[pos][h-1];if(!f||o){Code lsz=s.dp[pos][h-1];intervals_rec(s,pos-1,h-1,gbase,lbase,out);lbase+=lsz;}gbase+=gsz;}
    if(h<MAXW+1&&(!f||o))intervals_rec(s,pos-1,h+1,gbase,lbase,out);
}
static std::vector<Interval> make_intervals(const GroupSpec&s){std::vector<Interval>v;v.reserve(1024);intervals_rec(s,s.width-1,1,0,0,v);Code sum=0;for(auto const&i:v)sum+=i.len;if(sum!=s.size){std::cerr<<"interval size mismatch "<<sum<<" != "<<s.size<<"\n";std::exit(2);}return v;}

static Code interval_leaf_upper_rec(const GroupSpec& s,int pos,int h,Code memo[MAXW+1][MAXW+2],bool seen[MAXW+1][MAXW+2]){
    if(pos<0)return h==0?1:0;
    if(seen[pos][h])return memo[pos][h];
    seen[pos][h]=true;
    uint32_t lower=(pos==31)?0xffffffffu:((1u<<(pos+1))-1u);
    if((s.fixed&lower)==0)return memo[pos][h]=H_DP[pos+1][h]?1:0;
    bool f=(s.fixed>>pos)&1u,o=(s.occ>>pos)&1u;
    Code z=0;
    if(!f||!o)z+=interval_leaf_upper_rec(s,pos-1,h,memo,seen);
    if(!f||o){
        if(h>0)z+=interval_leaf_upper_rec(s,pos-1,h-1,memo,seen);
        if(h<MAXW+1)z+=interval_leaf_upper_rec(s,pos-1,h+1,memo,seen);
    }
    return memo[pos][h]=z;
}
static Code interval_leaf_upper(const GroupSpec& s){
    Code memo[MAXW+1][MAXW+2]{};bool seen[MAXW+1][MAXW+2]{};
    return interval_leaf_upper_rec(s,s.width-1,1,memo,seen);
}

struct PeerInterval{Code remote,local,len;uint32_t owner,pad;};
static std::vector<PeerInterval> make_peer_intervals(const GroupSpec&s,Code chunk,int ng,bool& use_interval){
    constexpr Code MIN_AVG_INTERVAL_ELEMS = 65536;
    Code est=interval_leaf_upper(s);
    // Every shard boundary can split at most one globally ordered interval.
    Code est_peer=est+Code(std::max(0,ng-1));
    use_interval = est_peer==0 || s.size >= est_peer*MIN_AVG_INTERVAL_ELEMS;
    if(!use_interval)return {};
    auto base=make_intervals(s);std::vector<PeerInterval>out;out.reserve(base.size()+ng);
    for(auto const&x:base){Code g=x.global,l=x.local,left=x.len;while(left){int owner=int(g/chunk);if(owner>=ng)owner=ng-1;Code shard0=Code(owner)*chunk;Code shard_end=(owner+1<ng)?shard0+chunk:~Code(0);Code take=left;if(shard_end!=~Code(0)&&g+take>shard_end)take=shard_end-g;Code remote=g-shard0;
            if(!out.empty()&&out.back().owner==(uint32_t)owner&&out.back().remote+out.back().len==remote&&out.back().local+out.back().len==l)out.back().len+=take;else out.push_back({remote,l,take,(uint32_t)owner,0});
            g+=take;l+=take;left-=take;}}
    // Interval I/O wins only when the canonical layout contains reasonably long
    // contiguous runs.  Low fixed bits can fragment a group into hundreds of
    // thousands of tiny runs; in that case rank/unrank gather/scatter is faster.
    use_interval = out.empty() || s.size >= Code(out.size()) * MIN_AVG_INTERVAL_ELEMS;
    if(!use_interval) return {};

    std::vector<PeerInterval> tiled;
    size_t nt=0;
    for(auto const& x:out) nt += size_t((x.len + IO_TILE_ELEMS - 1) / IO_TILE_ELEMS);
    tiled.reserve(nt);
    for(auto const& x:out){
        Code off=0;
        while(off<x.len){
            Code take=std::min<Code>(x.len-off,Code(IO_TILE_ELEMS));
            tiled.push_back({x.remote+off,x.local+off,take,x.owner,0});
            off+=take;
        }
    }
    return tiled;
}

static std::vector<int> window_candidates(int W,int hi,int lo){std::vector<int>v;for(int q=W-1;q>=0;--q)if(q<lo-1||q>hi)v.push_back(q);return v;}
static void window_masks(int W,int hi,int lo,const std::vector<int>&fp,uint32_t group,uint32_t&mf,uint32_t&mo,uint32_t&bf,uint32_t&bo){mf=mo=bf=bo=0;for(size_t i=0;i<fp.size();++i){int q=fp[i];bool one=(group>>i)&1u;mf|=1u<<q;if(one)mo|=1u<<q;int bq=(q<lo-1)?q:q-1;bf|=1u<<bq;if(one)bo|=1u<<bq;}}
struct WindowPlan{int p_hi=0,p_lo=0;std::vector<int>fixed_pos;size_t max_bytes=0;Code max_main=0,max_block=0;};
static WindowPlan plan_window(int W,int hi,int lo,size_t target,int maxbits=20){WindowPlan best;best.p_hi=hi;best.p_lo=lo;auto cand=window_candidates(W,hi,lo);int klim=std::min<int>(cand.size(),maxbits);for(int k=0;k<=klim;++k){std::vector<int>fp(cand.begin(),cand.begin()+k);uint64_t ng=1ull<<k;size_t mx=0;Code mm=0,md=0;for(uint64_t g=0;g<ng;++g){uint32_t mf,mo,bf,bo;window_masks(W,hi,lo,fp,(uint32_t)g,mf,mo,bf,bo);auto ms=make_spec(W,mf,mo);auto ds=make_spec(W-1,bf,bo);size_t b=size_t(2*ms.size+2*ds.size)*sizeof(Count);if(b>mx){mx=b;mm=ms.size;md=ds.size;}if(mx>target&&k<klim)break;}if(mx<=target||k==klim){best.fixed_pos=std::move(fp);best.max_bytes=mx;best.max_main=mm;best.max_block=md;return best;}}return best;}

__device__ __forceinline__ bool allowed(uint32_t fixed,uint32_t occ,int pos,MateValue v){if(!((fixed>>pos)&1u))return v!=X;bool o=(occ>>pos)&1u;return o?(v==R||v==L):(v==N);}
__device__ __forceinline__ MateID unrank_group(Code rank,int width,uint32_t fixed,uint32_t occ,const Code dp[MAXW+1][MAXW+2]){MateID m=0;int h=1;
#pragma unroll
    for(int pos=MAXW-1;pos>=0;--pos){if(pos>=width)continue;if(allowed(fixed,occ,pos,N)){Code z=dp[pos][h];if(rank<z)continue;rank-=z;}if(h>0&&allowed(fixed,occ,pos,R)){Code z=dp[pos][h-1];if(rank<z){m|=MateID(R)<<(2*pos);--h;continue;}rank-=z;}m|=MateID(L)<<(2*pos);++h;}return m;}
__device__ __forceinline__ Code rank_group(MateID m,int width,uint32_t fixed,uint32_t occ,const Code dp[MAXW+1][MAXW+2]){Code rank=0;int h=1;
#pragma unroll
    for(int pos=MAXW-1;pos>=0;--pos){if(pos>=width)continue;MateValue s=mget(m,pos);if(s>N&&allowed(fixed,occ,pos,N))rank+=dp[pos][h];if(s>R&&h>0&&allowed(fixed,occ,pos,R))rank+=dp[pos][h-1];if(s==R)--h;else if(s==L)++h;}return rank;}
__device__ __forceinline__ Code rank_full_dev(MateID m,int width){Code rank=0;int h=1;
#pragma unroll
    for(int pos=MAXW-1;pos>=0;--pos){if(pos>=width)continue;MateValue s=mget(m,pos);if(s>N)rank+=D_FULL_DP[pos][h];if(s>R&&h>0)rank+=D_FULL_DP[pos][h-1];if(s==R)--h;else if(s==L)++h;}return rank;}

template<int WIDTH> __device__ __forceinline__ Code rank_full_t(MateID m);

template<int WIDTH>
__device__ __forceinline__ bool high_lut_applicable(uint32_t fixed){
    if constexpr(HIGH_LUT_K==0)return false;
    else {
        constexpr int LOW=WIDTH-HIGH_LUT_K;
        constexpr uint32_t HM=((1u<<HIGH_LUT_K)-1u)<<LOW;
        constexpr uint32_t WM=(WIDTH==32)?0xffffffffu:((1u<<WIDTH)-1u);
        return LOW>=0 && (fixed&HM)==HM && (fixed&(WM^HM))==0;
    }
}

template<int WIDTH>
__device__ __forceinline__ int high_final_height(uint32_t code){
    int h=1;
#pragma unroll
    for(int p=HIGH_LUT_K-1;p>=0;--p){
        MateValue v=MateValue((code>>(2*p))&3u);
        if(v==R)--h; else if(v==L)++h;
    }
    return h;
}

template<int WIDTH>
__device__ __forceinline__ MateID unrank_free_suffix(Code rank,int width,int h){
    MateID m=0;
#pragma unroll
    for(int pos=WIDTH-1;pos>=0;--pos){
        if(pos>=width)continue;
        Code z=D_FULL_DP[pos][h];
        if(rank<z)continue;
        rank-=z;
        if(h>0){z=D_FULL_DP[pos][h-1];if(rank<z){m|=MateID(R)<<(2*pos);--h;continue;}rank-=z;}
        m|=MateID(L)<<(2*pos);++h;
    }
    return m;
}

template<int WIDTH>
__device__ __forceinline__ Code rank_free_suffix(MateID m,int width,int h){
    Code rank=0;
#pragma unroll
    for(int pos=WIDTH-1;pos>=0;--pos){
        if(pos>=width)continue;
        MateValue v=mget(m,pos);
        if(v>N)rank+=D_FULL_DP[pos][h];
        if(v>R&&h>0)rank+=D_FULL_DP[pos][h-1];
        if(v==R)--h;else if(v==L)++h;
    }
    return rank;
}

template<int WIDTH>
__device__ __forceinline__ MateID unrank_high_lut(Code rank,uint32_t occ){
    constexpr int LOW=WIDTH-HIGH_LUT_K;
    constexpr uint32_t OM=(1u<<HIGH_LUT_K)-1u;
    uint32_t mask=(occ>>LOW)&OM;
    HighEntry* entries = WIDTH==TARGET_W ? D_HIGH_ENTRIES_MAIN : D_HIGH_ENTRIES_BLOCK;
    uint32_t* offsets = WIDTH==TARGET_W ? D_HIGH_OFFSETS_MAIN : D_HIGH_OFFSETS_BLOCK;
    uint32_t a=offsets[mask], b=offsets[mask+1];
    // Find the last prefix whose base is <= rank.
    uint32_t lo=a,hi=b;
    while(lo+1<hi){uint32_t mid=(lo+hi)>>1;if(Code(entries[mid].base)<=rank)lo=mid;else hi=mid;}
    HighEntry e=entries[lo];
    int h=high_final_height<WIDTH>(e.code);
    MateID low=unrank_free_suffix<WIDTH>(rank-Code(e.base),LOW,h);
    return (MateID(e.code)<<(2*LOW))|low;
}

template<int WIDTH>
__device__ __forceinline__ Code rank_high_lut(MateID m,uint32_t occ){
    constexpr int LOW=WIDTH-HIGH_LUT_K;
    constexpr MateID CM=(MateID(1)<<(2*HIGH_LUT_K))-1;
    constexpr uint32_t OM=(1u<<HIGH_LUT_K)-1u;
    uint32_t code=(uint32_t)((m>>(2*LOW))&CM);
    uint32_t mask=(occ>>LOW)&OM;
    HighEntry* entries = WIDTH==TARGET_W ? D_HIGH_ENTRIES_MAIN : D_HIGH_ENTRIES_BLOCK;
    uint32_t* offsets = WIDTH==TARGET_W ? D_HIGH_OFFSETS_MAIN : D_HIGH_OFFSETS_BLOCK;
    uint32_t lo=offsets[mask],hi=offsets[mask+1];
    while(lo<hi){uint32_t mid=(lo+hi)>>1;if(entries[mid].code<code)lo=mid+1;else hi=mid;}
    HighEntry e=entries[lo];
    int h=high_final_height<WIDTH>(code);
    constexpr MateID LM=(MateID(1)<<(2*LOW))-1;
    return Code(e.base)+rank_free_suffix<WIDTH>(m&LM,LOW,h);
}
template<int WIDTH>
__device__ __forceinline__ bool low_lut_applicable(uint32_t fixed){
    if constexpr(LOW_LUT_K==0)return false;
    else return (fixed&((1u<<LOW_LUT_K)-1u))==((1u<<LOW_LUT_K)-1u);
}
template<int WIDTH>
__device__ __forceinline__ MateID unrank_group_t(Code rank,uint32_t fixed,uint32_t occ,const Code dp[MAXW+1][MAXW+2]){
    if constexpr(HIGH_LUT_K>0){if(high_lut_applicable<WIDTH>(fixed))return unrank_high_lut<WIDTH>(rank,occ);}
    MateID m=0;int h=1;
    if constexpr(LOW_LUT_K>0){if(low_lut_applicable<WIDTH>(fixed)){
#pragma unroll
        for(int pos=WIDTH-1;pos>=LOW_LUT_K;--pos){
            if(allowed(fixed,occ,pos,N)){Code z=dp[pos][h];if(rank<z)continue;rank-=z;}
            if(h>0&&allowed(fixed,occ,pos,R)){Code z=dp[pos][h-1];if(rank<z){m|=MateID(R)<<(2*pos);--h;continue;}rank-=z;}
            m|=MateID(L)<<(2*pos);++h;
        }
        uint32_t mask=occ&((1u<<LOW_LUT_K)-1u);uint32_t base=D_LOW_OFFSETS[size_t(mask)*(LOW_LUT_K+2)+h];m|=D_LOW_ENTRIES[base+(uint32_t)rank].mate;return m;
    }}
#pragma unroll
    for(int pos=WIDTH-1;pos>=0;--pos){
        if(allowed(fixed,occ,pos,N)){Code z=dp[pos][h];if(rank<z)continue;rank-=z;}
        if(h>0&&allowed(fixed,occ,pos,R)){Code z=dp[pos][h-1];if(rank<z){m|=MateID(R)<<(2*pos);--h;continue;}rank-=z;}
        m|=MateID(L)<<(2*pos);++h;
    }
    return m;
}
template<int WIDTH>
__device__ __forceinline__ Code rank_group_t(MateID m,uint32_t fixed,uint32_t occ,const Code dp[MAXW+1][MAXW+2]){
    if constexpr(HIGH_LUT_K>0){if(high_lut_applicable<WIDTH>(fixed))return rank_high_lut<WIDTH>(m,occ);}
    Code rank=0;int h=1;
    if constexpr(LOW_LUT_K>0){if(low_lut_applicable<WIDTH>(fixed)){
#pragma unroll
        for(int pos=WIDTH-1;pos>=LOW_LUT_K;--pos){MateValue v=mget(m,pos);if(v>N&&allowed(fixed,occ,pos,N))rank+=dp[pos][h];if(v>R&&h>0&&allowed(fixed,occ,pos,R))rank+=dp[pos][h-1];if(v==R)--h;else if(v==L)++h;}
        constexpr MateID CM=(MateID(1)<<(2*LOW_LUT_K))-1;uint32_t lr=D_LOW_LOCAL_RANK[(uint32_t)(m&CM)];return rank+lr;
    }}
#pragma unroll
    for(int pos=WIDTH-1;pos>=0;--pos){MateValue v=mget(m,pos);if(v>N&&allowed(fixed,occ,pos,N))rank+=dp[pos][h];if(v>R&&h>0&&allowed(fixed,occ,pos,R))rank+=dp[pos][h-1];if(v==R)--h;else if(v==L)++h;}
    return rank;
}
template<int WIDTH>
__device__ __forceinline__ int height_before_rank_pos(MateID m,int hi){
    constexpr MateID EVEN=0x5555555555555555ULL;
    constexpr MateID WM=(MateID(1)<<(2*WIDTH))-1ULL;
    MateID lower=(MateID(1)<<(2*(hi+1)))-1ULL;
    MateID pm=EVEN&WM&~lower;
    int nr=__popcll(m&pm), nl=__popcll((m>>1)&pm);
    return 1+nl-nr;
}
template<int WIDTH>
__device__ __forceinline__ Code rank_slice_t(MateID m,int hi,int lo,int h,uint32_t fixed,uint32_t occ,const Code dp[MAXW+1][MAXW+2]){
    Code rank=0;
    for(int pos=hi;pos>=lo;--pos){MateValue v=mget(m,pos);if(v>N&&allowed(fixed,occ,pos,N))rank+=dp[pos][h];if(v>R&&h>0&&allowed(fixed,occ,pos,R))rank+=dp[pos][h-1];if(v==R)--h;else if(v==L)++h;}
    return rank;
}
template<int WIDTH>
__device__ __forceinline__ Code rank_same_t(Code src_rank,MateID src,MateID dst,uint32_t fixed,uint32_t occ,const Code dp[MAXW+1][MAXW+2]){
    MateID diff=src^dst;
    int hi=(63-__clzll(diff))/2;
    int lo=(__ffsll((long long)diff)-1)/2;
    int h=height_before_rank_pos<WIDTH>(src,hi);
    Code a=rank_slice_t<WIDTH>(src,hi,lo,h,fixed,occ,dp);
    Code b=rank_slice_t<WIDTH>(dst,hi,lo,h,fixed,occ,dp);
    return b>=a?src_rank+(b-a):src_rank-(a-b);
}

template<int WIDTH>
__device__ __forceinline__ MateID unrank_group_global_t(Code rank,uint32_t fixed,uint32_t occ,const Code dp[MAXW+1][MAXW+2],Code& global_rank){
    if constexpr(LOW_LUT_K>0){if(low_lut_applicable<WIDTH>(fixed)){
        MateID m=0;int h=1;global_rank=0;
#pragma unroll
        for(int pos=WIDTH-1;pos>=LOW_LUT_K;--pos){
            bool chose=false;
            if(allowed(fixed,occ,pos,N)){Code z=dp[pos][h];if(rank<z){chose=true;}else rank-=z;}
            if(chose)continue;
            if(h>0&&allowed(fixed,occ,pos,R)){Code z=dp[pos][h-1];if(rank<z){global_rank+=D_FULL_DP[pos][h];m|=MateID(R)<<(2*pos);--h;continue;}rank-=z;}
            global_rank+=D_FULL_DP[pos][h]+(h>0?D_FULL_DP[pos][h-1]:0);m|=MateID(L)<<(2*pos);++h;
        }
        uint32_t mask=occ&((1u<<LOW_LUT_K)-1u);uint32_t base=D_LOW_OFFSETS[size_t(mask)*(LOW_LUT_K+2)+h];LowEntry e=D_LOW_ENTRIES[base+(uint32_t)rank];m|=e.mate;global_rank+=e.full_rank;return m;
    }}
    MateID m=unrank_group_t<WIDTH>(rank,fixed,occ,dp);global_rank=rank_full_t<WIDTH>(m);return m;
}
template<int WIDTH>
__device__ __forceinline__ Code rank_full_t(MateID m){
    Code rank=0;int h=1;
#pragma unroll
    for(int pos=WIDTH-1;pos>=0;--pos){MateValue v=mget(m,pos);if(v>N)rank+=D_FULL_DP[pos][h];if(v>R&&h>0)rank+=D_FULL_DP[pos][h-1];if(v==R)--h;else if(v==L)++h;}
    return rank;
}
__device__ __forceinline__ Count global_load_main(Code g){int o=int(g/D_MAIN_CHUNK);if(o>=D_NGPU)o=D_NGPU-1;return D_MAIN_PTR[o][g-Code(o)*D_MAIN_CHUNK];}
__device__ __forceinline__ Count global_load_block(Code g){int o=int(g/D_BLOCK_CHUNK);if(o>=D_NGPU)o=D_NGPU-1;return D_BLOCK_PTR[o][g-Code(o)*D_BLOCK_CHUNK];}
__device__ __forceinline__ void global_store_main(Code g,Count v){int o=int(g/D_MAIN_CHUNK);if(o>=D_NGPU)o=D_NGPU-1;D_MAIN_PTR[o][g-Code(o)*D_MAIN_CHUNK]=v;}
__device__ __forceinline__ void global_store_block(Code g,Count v){int o=int(g/D_BLOCK_CHUNK);if(o>=D_NGPU)o=D_NGPU-1;D_BLOCK_PTR[o][g-Code(o)*D_BLOCK_CHUNK]=v;}

__global__ void gather_main_kernel(Count*out,MateID*mates,Code n){Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;for(;i<n;i+=stride){Code g;MateID m=unrank_group_global_t<TARGET_W>(i,D_MAIN_FIXED,D_MAIN_OCC,D_MAIN_DP,g);out[i]=global_load_main(g);if(mates)mates[i]=m;}}
__global__ void gather_block_kernel(Count*out,Code n){Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;for(;i<n;i+=stride){Code g;unrank_group_global_t<TARGET_W-1>(i,D_BLOCK_FIXED,D_BLOCK_OCC,D_BLOCK_DP,g);out[i]=global_load_block(g);}}
__global__ void scatter_main_kernel(const Count*in,Code n){Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;for(;i<n;i+=stride){Code g;unrank_group_global_t<TARGET_W>(i,D_MAIN_FIXED,D_MAIN_OCC,D_MAIN_DP,g);global_store_main(g,in[i]);}}
__global__ void scatter_block_kernel(const Count*in,Code n){Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;for(;i<n;i+=stride){Code g;unrank_group_global_t<TARGET_W-1>(i,D_BLOCK_FIXED,D_BLOCK_OCC,D_BLOCK_DP,g);global_store_block(g,in[i]);}}


template<bool BLOCK,bool SCATTER>
__global__ void interval_io_kernel(Count*buf,const PeerInterval*iv,size_t niv){
    for(size_t k=blockIdx.x;k<niv;k+=gridDim.x){
        PeerInterval x=iv[k];
        Count*peer=(BLOCK?D_BLOCK_PTR[x.owner]:D_MAIN_PTR[x.owner])+x.remote;
        for(Code off=threadIdx.x;off<x.len;off+=blockDim.x){
            if constexpr(SCATTER) peer[off]=buf[x.local+off];
            else buf[x.local+off]=peer[off];
        }
    }
}
static int interval_blocks(size_t niv,int){return int(std::min<size_t>(65535,std::max<size_t>(1,niv)));}

__device__ __forceinline__ void atomic_add_mod(Count*p,Count v){if(!v)return;Count mod=D_MOD;Count old=atomicCAS(p,0u,0u);for(;;){Count neu=(old>=mod-v)?old-(mod-v):old+v;Count seen=atomicCAS(p,old,neu);if(seen==old)return;old=seen;}}
template<int WIDTH>
__device__ __forceinline__ Code rank_drop_n_t(Code src_rank,MateID m,int p){Code a=0,b=0;int h=1;
#pragma unroll
for(int pos=WIDTH-1;pos>p;--pos){MateValue v=mget(m,pos);if(v>N&&allowed(D_MAIN_FIXED,D_MAIN_OCC,pos,N))a+=D_MAIN_DP[pos][h];if(v>R&&h>0&&allowed(D_MAIN_FIXED,D_MAIN_OCC,pos,R))a+=D_MAIN_DP[pos][h-1];int q=pos-1;if(v>N&&allowed(D_BLOCK_FIXED,D_BLOCK_OCC,q,N))b+=D_BLOCK_DP[q][h];if(v>R&&h>0&&allowed(D_BLOCK_FIXED,D_BLOCK_OCC,q,R))b+=D_BLOCK_DP[q][h-1];if(v==R)--h;else if(v==L)++h;}return b>=a?src_rank+(b-a):src_rank-(a-b);}
__global__ void blocked_group_kernel(const Count*in,Code n,Count*out_main,int p){Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;for(;i<n;i+=stride){Count c=in[i];if(!c)continue;MateID sm=unrank_group_t<TARGET_W-1>(i,D_BLOCK_FIXED,D_BLOCK_OCC,D_BLOCK_DP);MateID t=minsert(sm,p,N);Code j=rank_group_t<TARGET_W>(t,D_MAIN_FIXED,D_MAIN_OCC,D_MAIN_DP);atomic_add_mod(out_main+j,c);}}
__global__ void main_group_kernel(const Count*in,const MateID*mates,Code n,Count*out_main,Count*out_block,int p){Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;for(;i<n;i+=stride){Count c=in[i];if(!c)continue;MateID m=mates?mates[i]:unrank_group_t<TARGET_W>(i,D_MAIN_FIXED,D_MAIN_OCC,D_MAIN_DP);MateValuePair w=mpair(m,p);switch(w){case NN:{MateID t=msetpair(m,p,LR);atomic_add_mod(out_main+rank_same_t<TARGET_W>(i,m,t,D_MAIN_FIXED,D_MAIN_OCC,D_MAIN_DP),c);break;}case NR:case NL:{if(p==1){MateID t=msetpair(m,p,w==NR?RN:LN);atomic_add_mod(out_main+rank_same_t<TARGET_W>(i,m,t,D_MAIN_FIXED,D_MAIN_OCC,D_MAIN_DP),c);}else{Code j=rank_drop_n_t<TARGET_W>(i,m,p);atomic_add_mod(out_block+j,c);}break;}case RN:{MateID t=msetpair(m,p,NR);atomic_add_mod(out_main+rank_same_t<TARGET_W>(i,m,t,D_MAIN_FIXED,D_MAIN_OCC,D_MAIN_DP),c);break;}case LN:{MateID t=msetpair(m,p,NL);atomic_add_mod(out_main+rank_same_t<TARGET_W>(i,m,t,D_MAIN_FIXED,D_MAIN_OCC,D_MAIN_DP),c);break;}case LL:{MateID t=msetpair(m,p,NN);int q=p-1,s=1;while(s){--q;auto v=mget(t,q);if(v==L)++s;else if(v==R)--s;}t=mset(t,q,L);if(p==1)atomic_add_mod(out_main+rank_same_t<TARGET_W>(i,m,t,D_MAIN_FIXED,D_MAIN_OCC,D_MAIN_DP),c);else{t=mshrink(t,p-1);atomic_add_mod(out_block+rank_group_t<TARGET_W-1>(t,D_BLOCK_FIXED,D_BLOCK_OCC,D_BLOCK_DP),c);}break;}case RR:{MateID t=msetpair(m,p,NN);int q=p,s=1;while(s){++q;auto v=mget(t,q);if(v==L)--s;else if(v==R)++s;}t=mset(t,q,R);if(p==1)atomic_add_mod(out_main+rank_same_t<TARGET_W>(i,m,t,D_MAIN_FIXED,D_MAIN_OCC,D_MAIN_DP),c);else{t=mshrink(t,p-1);atomic_add_mod(out_block+rank_group_t<TARGET_W-1>(t,D_BLOCK_FIXED,D_BLOCK_OCC,D_BLOCK_DP),c);}break;}case RL:{MateID t=msetpair(m,p,NN);if(p==1)atomic_add_mod(out_main+rank_same_t<TARGET_W>(i,m,t,D_MAIN_FIXED,D_MAIN_OCC,D_MAIN_DP),c);else{t=mshrink(t,p-1);atomic_add_mod(out_block+rank_group_t<TARGET_W-1>(t,D_BLOCK_FIXED,D_BLOCK_OCC,D_BLOCK_DP),c);}break;}default:break;}}}

static Code rank_full(MateID m,int width){Code r=0;int h=1;for(int pos=width-1;pos>=0;--pos){auto s=mget(m,pos);if(s>N)r+=H_DP[pos][h];if(s>R&&h>0)r+=H_DP[pos][h-1];if(s==R)--h;else if(s==L)++h;}return r;}

struct DeviceCtx{
    int dev=-1;uint8_t*arena=nullptr;size_t capArena=0;Count*dA=nullptr,*dB=nullptr,*dD=nullptr,*dE=nullptr;MateID*dMate=nullptr;PeerInterval*dIM=nullptr,*dID=nullptr;size_t capIM=0,capID=0,maxIntervals=0;double active=0;uint64_t groups=0;cudaStream_t sMain=nullptr,sBlock=nullptr;cudaEvent_t copyDone=nullptr,clearDone=nullptr,mainDone=nullptr,blockDone=nullptr;
    void init(int d,Count mod,Count**mp,Count**bp,Code mc,Code bc,int ng){dev=d;ck(cudaSetDevice(dev),"set init");ck(cudaMemcpyToSymbol(D_FULL_DP,H_DP,sizeof(H_DP)),"full dp");ck(cudaMemcpyToSymbol(D_MOD,&mod,sizeof(mod)),"mod");ck(cudaMemcpyToSymbol(D_MAIN_PTR,mp,sizeof(Count*)*MAXGPU),"main ptrs");ck(cudaMemcpyToSymbol(D_BLOCK_PTR,bp,sizeof(Count*)*MAXGPU),"block ptrs");ck(cudaMemcpyToSymbol(D_MAIN_CHUNK,&mc,sizeof(mc)),"main chunk");ck(cudaMemcpyToSymbol(D_BLOCK_CHUNK,&bc,sizeof(bc)),"block chunk");ck(cudaMemcpyToSymbol(D_NGPU,&ng,sizeof(ng)),"ngpu");ck(cudaStreamCreateWithFlags(&sMain,cudaStreamNonBlocking),"stream main");ck(cudaStreamCreateWithFlags(&sBlock,cudaStreamNonBlocking),"stream block");ck(cudaEventCreateWithFlags(&copyDone,cudaEventDisableTiming),"event copy");ck(cudaEventCreateWithFlags(&clearDone,cudaEventDisableTiming),"event clear");ck(cudaEventCreateWithFlags(&mainDone,cudaEventDisableTiming),"event main");ck(cudaEventCreateWithFlags(&blockDone,cudaEventDisableTiming),"event block");}
    void ensure(Code m,Code b,bool useMate,size_t im,size_t id){ck(cudaSetDevice(dev),"set ensure");auto al=[](size_t x){return(x+255)&~size_t(255);};size_t ab=al(size_t(m)*sizeof(Count)),db=al(size_t(b)*sizeof(Count)),mb=useMate?al(size_t(m)*sizeof(MateID)):0,need=2*ab+2*db+mb;if(need>capArena){if(arena)cudaFree(arena);capArena=need;ck(cudaMalloc(&arena,capArena),"scratch arena");}size_t off=0;dA=(Count*)(arena+off);off+=ab;dB=(Count*)(arena+off);off+=ab;dD=(Count*)(arena+off);off+=db;dE=(Count*)(arena+off);off+=db;dMate=useMate?(MateID*)(arena+off):nullptr;if(im>capIM){if(dIM)cudaFree(dIM);capIM=im;ck(cudaMalloc(&dIM,capIM*sizeof(PeerInterval)),"interval main");}if(id>capID){if(dID)cudaFree(dID);capID=id;ck(cudaMalloc(&dID,capID*sizeof(PeerInterval)),"interval block");}maxIntervals=std::max({maxIntervals,im,id});}
    void destroy(){if(dev<0)return;cudaSetDevice(dev);if(arena)cudaFree(arena);if(dIM)cudaFree(dIM);if(dID)cudaFree(dID);if(copyDone)cudaEventDestroy(copyDone);if(clearDone)cudaEventDestroy(clearDone);if(mainDone)cudaEventDestroy(mainDone);if(blockDone)cudaEventDestroy(blockDone);if(sMain)cudaStreamDestroy(sMain);if(sBlock)cudaStreamDestroy(sBlock);}
};

struct PreparedGroup{
    int g=0;
    uint32_t mf=0,mo=0,bf=0,bo=0;
    GroupSpec ms,ds;
    bool use_mi=false,use_di=false;
    std::vector<PeerInterval> mi,di;
    Code work=0;
};
struct PreparedWindow{
    WindowPlan wp;
    std::vector<PreparedGroup> groups;
};

static PreparedGroup prepare_group_masks(int W,uint32_t mf,uint32_t mo,uint32_t bf,uint32_t bo,Code mc,Code bc,int ng){
    PreparedGroup pg;pg.mf=mf;pg.mo=mo;pg.bf=bf;pg.bo=bo;
    pg.ms=make_spec(W,mf,mo);pg.ds=make_spec(W-1,bf,bo);
    pg.work=2*pg.ms.size+pg.ds.size;
    pg.mi=make_peer_intervals(pg.ms,mc,ng,pg.use_mi);
    pg.di=make_peer_intervals(pg.ds,bc,ng,pg.use_di);
    return pg;
}
static PreparedGroup prepare_group(int W,const WindowPlan&wp,int g,Code mc,Code bc,int ng){
    uint32_t mf,mo,bf,bo;window_masks(W,wp.p_hi,wp.p_lo,wp.fixed_pos,(uint32_t)g,mf,mo,bf,bo);
    auto pg=prepare_group_masks(W,mf,mo,bf,bo,mc,bc,ng);pg.g=g;return pg;
}
static std::vector<int> lut_base_fixed_positions(int W,int hi,int lo){
    std::vector<int> fp;
    if constexpr(BASE_LUT_K==0) return fp;
    int low_free=lo-1, high_free=W-1-hi;
    if(low_free>=BASE_LUT_K){for(int q=0;q<BASE_LUT_K;++q)fp.push_back(q);return fp;}
    if(high_free>=BASE_LUT_K){for(int q=W-BASE_LUT_K;q<W;++q)fp.push_back(q);return fp;}
    return {};
}
static bool adaptive_refine_group(int W,const WindowPlan&wp,const PreparedGroup&pg,size_t target,Code mc,Code bc,int ng,std::vector<PreparedGroup>&out){
    size_t cb=size_t(2*pg.ms.size+2*pg.ds.size)*sizeof(Count);
    size_t mb=size_t(pg.ms.size)*sizeof(MateID);
    if(cb<=target && (pg.use_mi || cb+mb<=target)){out.push_back(pg);return true;}
    auto cand=window_candidates(W,wp.p_hi,wp.p_lo);
    int q=-1;for(int x:cand)if(((pg.mf>>x)&1u)==0){q=x;break;}
    if(q<0){if(cb<=target){out.push_back(pg);return true;}return false;}
    int bq=(q<wp.p_lo-1)?q:q-1;
    for(int one=0;one<2;++one){
        uint32_t mf=pg.mf|(1u<<q),mo=pg.mo,bf=pg.bf|(1u<<bq),bo=pg.bo;
        if(one){mo|=1u<<q;bo|=1u<<bq;}
        auto ch=prepare_group_masks(W,mf,mo,bf,bo,mc,bc,ng);
        if(!adaptive_refine_group(W,wp,ch,target,mc,bc,ng,out))return false;
    }
    return true;
}

static void process_group(DeviceCtx&c,int W,const WindowPlan&wp,const PreparedGroup&pg,int threads,size_t target){
    auto t0=std::chrono::steady_clock::now();ck(cudaSetDevice(c.dev),"set worker");
    auto const&ms=pg.ms;auto const&ds=pg.ds;if(!ms.size&&!ds.size)return;
    size_t countBytes=size_t(2*ms.size+2*ds.size)*sizeof(Count),mateBytes=size_t(ms.size)*sizeof(MateID);bool useMate=!pg.use_mi&&(countBytes+mateBytes<=target);
    c.ensure(ms.size,ds.size,useMate,pg.mi.size(),pg.di.size());
    if(!pg.mi.empty())ck(cudaMemcpy(c.dIM,pg.mi.data(),pg.mi.size()*sizeof(PeerInterval),cudaMemcpyHostToDevice),"copy main intervals");
    if(!pg.di.empty())ck(cudaMemcpy(c.dID,pg.di.data(),pg.di.size()*sizeof(PeerInterval),cudaMemcpyHostToDevice),"copy block intervals");
    ck(cudaMemcpyToSymbol(D_MAIN_DP,ms.dp,sizeof(ms.dp)),"main dp");ck(cudaMemcpyToSymbol(D_BLOCK_DP,ds.dp,sizeof(ds.dp)),"block dp");
    ck(cudaMemcpyToSymbol(D_MAIN_FIXED,&pg.mf,sizeof(pg.mf)),"mf");ck(cudaMemcpyToSymbol(D_MAIN_OCC,&pg.mo,sizeof(pg.mo)),"mo");
    ck(cudaMemcpyToSymbol(D_BLOCK_FIXED,&pg.bf,sizeof(pg.bf)),"bf");ck(cudaMemcpyToSymbol(D_BLOCK_OCC,&pg.bo,sizeof(pg.bo)),"bo");int mw=W,bw=W-1;ck(cudaMemcpyToSymbol(D_MAIN_W,&mw,sizeof(mw)),"mw");ck(cudaMemcpyToSymbol(D_BLOCK_W,&bw,sizeof(bw)),"bw");
    int bm=int(std::min<Code>(65535,(ms.size+threads-1)/threads)),bd=int(std::min<Code>(65535,(ds.size+threads-1)/threads));
    if(ms.size){if(pg.use_mi)interval_io_kernel<false,false><<<interval_blocks(pg.mi.size(),threads),threads>>>(c.dA,c.dIM,pg.mi.size());else gather_main_kernel<<<bm,threads>>>(c.dA,useMate?c.dMate:nullptr,ms.size);}
    if(ds.size){if(pg.use_di)interval_io_kernel<true,false><<<interval_blocks(pg.di.size(),threads),threads>>>(c.dD,c.dID,pg.di.size());else gather_block_kernel<<<bd,threads>>>(c.dD,ds.size);}
    ck(cudaGetLastError(),"doubleD gather");ck(cudaDeviceSynchronize(),"doubleD gather sync");
    Count*cur=c.dA,*nxt=c.dB,*dcur=c.dD,*dnext=c.dE;
    for(int p=wp.p_hi;p>=wp.p_lo;--p){
        if(ms.size)ck(cudaMemcpyAsync(nxt,cur,size_t(ms.size)*sizeof(Count),cudaMemcpyDeviceToDevice,c.sMain),"identity async");
        if(ds.size)ck(cudaMemsetAsync(dnext,0,size_t(ds.size)*sizeof(Count),c.sBlock),"clear next D");
        ck(cudaEventRecord(c.copyDone,c.sMain),"record copy");ck(cudaEventRecord(c.clearDone,c.sBlock),"record clear");
        ck(cudaStreamWaitEvent(c.sMain,c.clearDone,0),"main wait clear");ck(cudaStreamWaitEvent(c.sBlock,c.copyDone,0),"block wait copy");
        if(ms.size)main_group_kernel<<<bm,threads,0,c.sMain>>>(cur,useMate?c.dMate:nullptr,ms.size,nxt,dnext,p);
        if(ds.size)blocked_group_kernel<<<bd,threads,0,c.sBlock>>>(dcur,ds.size,nxt,p);
        ck(cudaGetLastError(),"doubleD transition");
        ck(cudaEventRecord(c.mainDone,c.sMain),"record main");ck(cudaEventRecord(c.blockDone,c.sBlock),"record block");
        ck(cudaStreamWaitEvent(c.sMain,c.blockDone,0),"main wait block");ck(cudaStreamWaitEvent(c.sBlock,c.mainDone,0),"block wait main");
        std::swap(cur,nxt);std::swap(dcur,dnext);
    }
    ck(cudaStreamSynchronize(c.sMain),"main sync");ck(cudaStreamSynchronize(c.sBlock),"block sync");
    if(ms.size){if(pg.use_mi)interval_io_kernel<false,true><<<interval_blocks(pg.mi.size(),threads),threads>>>(cur,c.dIM,pg.mi.size());else scatter_main_kernel<<<bm,threads>>>(cur,ms.size);}
    if(ds.size){if(pg.use_di)interval_io_kernel<true,true><<<interval_blocks(pg.di.size(),threads),threads>>>(dcur,c.dID,pg.di.size());else scatter_block_kernel<<<bd,threads>>>(dcur,ds.size);}
    ck(cudaGetLastError(),"doubleD scatter");ck(cudaDeviceSynchronize(),"group sync");c.groups++;c.active+=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
}

int main(int argc,char**argv){
    int n=argc>1?std::atoi(argv[1]):16;
    int target_mib=argc>2?std::atoi(argv[2]):16384;
    int max_window=argc>3?std::atoi(argv[3]):14;
    int requested=argc>4?std::atoi(argv[4]):0;
    std::vector<Count> mods;
    for(int i=5;i<argc;++i){
        unsigned long long raw=std::strtoull(argv[i],nullptr,10);
        if(raw<2||raw>0xffffffffULL){std::cerr<<"HBM32 modulus must be in [2, 4294967295], got "<<raw<<"\n";return 1;}
        mods.push_back((Count)raw);
    }
    if(mods.empty())mods.push_back(4294967291u);
    int W=n+1;
    if(n<2||W>MAXW){std::cerr<<"n=2..27\n";return 1;}
    if(W!=TARGET_W){std::cerr<<"specialized for width "<<TARGET_W<<" (n="<<(TARGET_W-1)<<")\n";return 1;}
    build_full_dp();
    int visible=0;ck(cudaGetDeviceCount(&visible),"count");
    int ng=requested<=0?visible:std::min(requested,visible);
    if(ng<1||ng>MAXGPU){std::cerr<<"need 1..8 GPUs\n";return 2;}
    int peers=0;
    for(int a=0;a<ng;++a)for(int b=0;b<ng;++b)if(a!=b){int can=0;ck(cudaDeviceCanAccessPeer(&can,a,b),"can peer");if(can){cudaSetDevice(a);auto e=cudaDeviceEnablePeerAccess(b,0);if(e==cudaErrorPeerAccessAlreadyEnabled)cudaGetLastError();else ck(e,"enable peer");peers++;}}
    if(ng>1&&peers!=ng*(ng-1)){std::cerr<<"HBM mode requires full P2P: "<<peers<<"/"<<ng*(ng-1)<<"\n";return 3;}

    LowTablesHost low=build_low_tables();LowEntry* lowE[MAXGPU]{};uint32_t* lowO[MAXGPU]{},*lowR[MAXGPU]{};
    if constexpr(LOW_LUT_K>0){for(int d=0;d<ng;++d){cudaSetDevice(d);ck(cudaMalloc(&lowE[d],low.entries.size()*sizeof(LowEntry)),"low entries");ck(cudaMemcpy(lowE[d],low.entries.data(),low.entries.size()*sizeof(LowEntry),cudaMemcpyHostToDevice),"low entries copy");ck(cudaMalloc(&lowO[d],low.offsets.size()*sizeof(uint32_t)),"low offsets");ck(cudaMemcpy(lowO[d],low.offsets.data(),low.offsets.size()*sizeof(uint32_t),cudaMemcpyHostToDevice),"low offsets copy");ck(cudaMalloc(&lowR[d],low.local_rank.size()*sizeof(uint32_t)),"low rank");ck(cudaMemcpy(lowR[d],low.local_rank.data(),low.local_rank.size()*sizeof(uint32_t),cudaMemcpyHostToDevice),"low rank copy");ck(cudaMemcpyToSymbol(D_LOW_ENTRIES,&lowE[d],sizeof(lowE[d])),"low entries ptr");ck(cudaMemcpyToSymbol(D_LOW_OFFSETS,&lowO[d],sizeof(lowO[d])),"low offsets ptr");ck(cudaMemcpyToSymbol(D_LOW_LOCAL_RANK,&lowR[d],sizeof(lowR[d])),"low rank ptr");}}
    HighTablesHost highM=build_high_tables(W),highB=build_high_tables(W-1);HighEntry* highEM[MAXGPU]{},*highEB[MAXGPU]{};uint32_t* highOM[MAXGPU]{},*highOB[MAXGPU]{};
    if constexpr(HIGH_LUT_K>0){for(int d=0;d<ng;++d){cudaSetDevice(d);ck(cudaMalloc(&highEM[d],highM.entries.size()*sizeof(HighEntry)),"high main entries");ck(cudaMemcpy(highEM[d],highM.entries.data(),highM.entries.size()*sizeof(HighEntry),cudaMemcpyHostToDevice),"high main entries copy");ck(cudaMalloc(&highEB[d],highB.entries.size()*sizeof(HighEntry)),"high block entries");ck(cudaMemcpy(highEB[d],highB.entries.data(),highB.entries.size()*sizeof(HighEntry),cudaMemcpyHostToDevice),"high block entries copy");ck(cudaMalloc(&highOM[d],highM.offsets.size()*sizeof(uint32_t)),"high main offsets");ck(cudaMemcpy(highOM[d],highM.offsets.data(),highM.offsets.size()*sizeof(uint32_t),cudaMemcpyHostToDevice),"high main offsets copy");ck(cudaMalloc(&highOB[d],highB.offsets.size()*sizeof(uint32_t)),"high block offsets");ck(cudaMemcpy(highOB[d],highB.offsets.data(),highB.offsets.size()*sizeof(uint32_t),cudaMemcpyHostToDevice),"high block offsets copy");ck(cudaMemcpyToSymbol(D_HIGH_ENTRIES_MAIN,&highEM[d],sizeof(highEM[d])),"high main entries ptr");ck(cudaMemcpyToSymbol(D_HIGH_ENTRIES_BLOCK,&highEB[d],sizeof(highEB[d])),"high block entries ptr");ck(cudaMemcpyToSymbol(D_HIGH_OFFSETS_MAIN,&highOM[d],sizeof(highOM[d])),"high main offsets ptr");ck(cudaMemcpyToSymbol(D_HIGH_OFFSETS_BLOCK,&highOB[d],sizeof(highOB[d])),"high block offsets ptr");}}

    Code mainN=H_DP[W][1],blockN=H_DP[W-1][1];Code mc=(mainN+ng-1)/ng,bc=(blockN+ng-1)/ng;Count*mp[MAXGPU]{},*bp[MAXGPU]{};std::vector<Code>ml(ng),bl(ng);
    for(int d=0;d<ng;++d){ml[d]=std::min<Code>(mc,mainN-std::min<Code>(mainN,Code(d)*mc));bl[d]=std::min<Code>(bc,blockN-std::min<Code>(blockN,Code(d)*bc));cudaSetDevice(d);if(ml[d])ck(cudaMalloc(&mp[d],size_t(ml[d])*sizeof(Count)),"auth main");if(bl[d])ck(cudaMalloc(&bp[d],size_t(bl[d])*sizeof(Count)),"auth block");}
    std::vector<DeviceCtx>ctx(ng);for(int d=0;d<ng;++d)ctx[d].init(d,mods[0],mp,bp,mc,bc,ng);
    size_t min_free=~size_t(0),min_total=~size_t(0);for(int d=0;d<ng;++d){ck(cudaSetDevice(d),"set meminfo");size_t f=0,t=0;ck(cudaMemGetInfo(&f,&t),"cudaMemGetInfo");min_free=std::min(min_free,f);min_total=std::min(min_total,t);}
    int reserve_mib=std::min(8192,std::max(256,int((min_total>>20)/32)));if(const char*e=std::getenv("GRIDFP_VRAM_RESERVE_MIB")){int v=std::atoi(e);if(v>=0)reserve_mib=v;}
    size_t requested_target=size_t(std::max(1,target_mib))<<20;size_t reserve=size_t(reserve_mib)<<20;if(min_free<=reserve+(64ull<<20)){std::cerr<<"insufficient HBM after authoritative state: min_free_mib="<<(min_free>>20)<<" reserve_mib="<<reserve_mib<<"\n";return 5;}
    size_t target=std::min(requested_target,min_free-reserve);int effective_target_mib=int(target>>20);
    std::cerr<<"HBM32 batch memory: auth_gib="<<double(mainN+blockN)*sizeof(Count)/(1ull<<30)<<" auth_per_gpu_gib="<<double(mainN+blockN)*sizeof(Count)/ng/(1ull<<30)<<" min_total_gib="<<double(min_total)/(1ull<<30)<<" min_free_after_auth_gib="<<double(min_free)/(1ull<<30)<<" requested_scratch_mib="<<target_mib<<" effective_scratch_mib="<<effective_target_mib<<" reserve_mib="<<reserve_mib<<" moduli="<<mods.size()<<"\n";

    int threads=256,maxgroups=0;auto prep0=std::chrono::steady_clock::now();std::vector<PreparedWindow> schedule;
    for(int hi=W-1;hi>=1;){
        PreparedWindow chosen;bool found=false;
        for(int lo=std::max(1,hi-max_window+1);lo<=hi;++lo){
            WindowPlan wp;wp.p_hi=hi;wp.p_lo=lo;wp.fixed_pos=lut_base_fixed_positions(W,hi,lo);
            if constexpr(BASE_LUT_K>0){if((int)wp.fixed_pos.size()!=BASE_LUT_K)continue;}
            int k=wp.fixed_pos.size();int nj=1<<k;PreparedWindow pw;pw.wp=wp;pw.groups.reserve(nj+32);bool ok=true;
            for(int g=0;g<nj;++g){auto pg=prepare_group(W,pw.wp,g,mc,bc,ng);if(!adaptive_refine_group(W,pw.wp,pg,target,mc,bc,ng,pw.groups)){ok=false;break;}}
            if(ok){chosen=std::move(pw);found=true;break;}
        }
        if(!found){std::cerr<<"cannot fit adaptive LUT window hi="<<hi<<"\n";return 4;}
        maxgroups=std::max(maxgroups,(int)chosen.groups.size());std::sort(chosen.groups.begin(),chosen.groups.end(),[](auto const&a,auto const&b){return a.work>b.work;});int next_hi=chosen.wp.p_lo-1;schedule.push_back(std::move(chosen));hi=next_hi;
    }
    double prepare_s=std::chrono::duration<double>(std::chrono::steady_clock::now()-prep0).count();std::cerr<<"prepared windows="<<schedule.size()<<" max_groups="<<maxgroups<<" prepare_s="<<prepare_s<<"\n";

    MateID init=MateID(R)<<(2*(W-1));Code ig=rank_full(init,W);int io=int(ig/mc);Code fg=rank_full(MateID(R),W);int fo=int(fg/mc);Count one=1;
    for(size_t ri=0;ri<mods.size();++ri){Count mod=mods[ri];
        for(int d=0;d<ng;++d){ck(cudaSetDevice(d),"set residue reset");ck(cudaMemcpyToSymbol(D_MOD,&mod,sizeof(mod)),"set modulus");if(ml[d])ck(cudaMemset(mp[d],0,size_t(ml[d])*sizeof(Count)),"zero main");if(bl[d])ck(cudaMemset(bp[d],0,size_t(bl[d])*sizeof(Count)),"zero block");ck(cudaDeviceSynchronize(),"zero sync");ctx[d].active=0;ctx[d].groups=0;}
        ck(cudaSetDevice(io),"set init device");ck(cudaMemcpy(mp[io]+(ig-Code(io)*mc),&one,sizeof(one),cudaMemcpyHostToDevice),"init one");
        auto wall0=std::chrono::steady_clock::now();int done_windows=0;
        for(int row=0;row<W;++row){for(auto const&pw:schedule){int nj=(int)pw.groups.size();std::atomic<int>next{0};std::vector<std::thread>ths;ths.reserve(ng);for(int d=0;d<ng;++d)ths.emplace_back([&,d]{for(;;){int q=next.fetch_add(1,std::memory_order_relaxed);if(q>=nj)break;process_group(ctx[d],W,pw.wp,pw.groups[q],threads,target);}});for(auto&t:ths)t.join();++done_windows;}std::cerr<<"mod "<<(ri+1)<<"/"<<mods.size()<<" p="<<mod<<" row "<<row+1<<"/"<<W<<"\n";}
        double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-wall0).count();Count ans=0;ck(cudaSetDevice(fo),"set answer device");ck(cudaMemcpy(&ans,mp[fo]+(fg-Code(fo)*mc),sizeof(ans),cudaMemcpyDeviceToHost),"answer");double mx=0,sum=0;size_t maxIntervals=0;for(auto&c:ctx){mx=std::max(mx,c.active);sum+=c.active;maxIntervals=std::max(maxIntervals,c.maxIntervals);}std::cout<<"backend=gridfp-b300-hbm32-adaptive-lutbase-batch n="<<n<<" residue="<<ans<<" modulus="<<mod<<" residue_index="<<ri<<" residues_total="<<mods.size()<<" gpus="<<ng<<" peers="<<peers<<" main_states="<<mainN<<" blocked_states="<<blockN<<" scratch_target_mib="<<effective_target_mib<<" windows="<<done_windows<<" max_groups="<<maxgroups<<" max_intervals="<<maxIntervals<<" active_max_s="<<mx<<" active_sum_s="<<sum<<" prepare_s="<<prepare_s<<" wall_s="<<wall<<std::endl;
    }

    for(auto&c:ctx)c.destroy();for(int d=0;d<ng;++d){cudaSetDevice(d);if(mp[d])cudaFree(mp[d]);if(bp[d])cudaFree(bp[d]);if(lowE[d])cudaFree(lowE[d]);if(lowO[d])cudaFree(lowO[d]);if(lowR[d])cudaFree(lowR[d]);if(highEM[d])cudaFree(highEM[d]);if(highEB[d])cudaFree(highEB[d]);if(highOM[d])cudaFree(highOM[d]);if(highOB[d])cudaFree(highOB[d]);}
}
