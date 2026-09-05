#include <cuda_runtime.h>
#include <cuda/atomic>
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
#include <functional>
#include <vector>
#include "../../common/gridfp_transition.hpp"
#include "row3_automaton_generated.hpp"
#include "row4_automaton_mod4294967291.hpp"
#include "row5_automaton_mod4294967291.hpp"
#include "row5_automaton_rational.hpp"
#include "row6_automaton_mod1000000007.hpp"

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

enum MateValue:uint8_t{N=0,R=1,L=2,X=3};
enum MateValuePair:uint8_t{NN=0x0,NR=0x1,NL=0x2,NX=0x3,RN=0x4,RR=0x5,RL=0x6,RX=0x7,LN=0x8,LR=0x9,LL=0xa,LX=0xb,XN=0xc,XR=0xd,XL=0xe,XX=0xf};

static Code H_DP[MAXW+1][MAXW+2];
__constant__ Code D_FULL_DP[MAXW+1][MAXW+2];
__constant__ Code D_BOUND_DP[MAXW+1][MAXW+2];
__constant__ Code D_BOUND_OLD_DP[MAXW+1][MAXW+2];
__constant__ int D_BOUND_CAP,D_BOUND_OLD_CAP;
__constant__ Code D_MAIN_DP[MAXW+1][MAXW+2];
__constant__ Code D_BLOCK_DP[MAXW+1][MAXW+2];
__constant__ uint32_t D_MAIN_FIXED,D_MAIN_OCC,D_BLOCK_FIXED,D_BLOCK_OCC;
__constant__ int D_MAIN_W,D_BLOCK_W,D_NGPU;
__constant__ Count D_MOD;
__constant__ Count* D_R3_PREF;
__constant__ Count* D_R3_SUFF;
__constant__ Count D_R3_INV_SCALE;
__constant__ Count* D_R4_PREF;
__constant__ Count* D_R4_SUFF;
__constant__ Count* D_R5_PREF;
__constant__ Count* D_R5_SUFF;
__constant__ uint32_t* D_R5_PREF_IDX;
__constant__ uint32_t* D_R5_SUFF_IDX;
__constant__ Count* D_R6_PREF;
__constant__ Count* D_R6_SUFF;
__constant__ uint32_t* D_R6_PREF_IDX;
__constant__ uint32_t* D_R6_SUFF_IDX;
__constant__ uint32_t D_R6_GPU_MAT_OFF[3][7];
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

__constant__ uint32_t* D_F_LOW_ALL_CODES;
__constant__ uint32_t* D_F_LOW_MASK_CODES;
__constant__ uint32_t* D_F_LOW_MASK_OFF;
__constant__ uint32_t* D_F_LOW_PACKED_RANK;
__constant__ uint32_t* D_F_HIGH_ALL_CODES;
__constant__ uint32_t* D_F_HIGH_MASK_CODES;
__constant__ uint32_t* D_F_HIGH_MASK_OFF;
__constant__ uint32_t* D_F_HIGH_PACKED_RANK;
__constant__ uint16_t* D_TRIT7_PTR;
__constant__ Code* D_F_HIGH_MAIN_BASE;
__constant__ Code* D_F_HIGH_BLOCK_BASE;
__constant__ uint32_t D_F_LOW_ALL_OFF[MAXW+2];
__constant__ uint32_t D_F_HIGH_ALL_OFF[MAXW+2];
__constant__ uint32_t D_F_MASK;
__constant__ int D_F_FIX_LOW;
struct FBlock { Code off,end; uint32_t stride; uint8_t he,hs,c,pad; };
__constant__ FBlock D_F_MAIN_BLOCKS[64];
__constant__ FBlock D_F_BLOCK_BLOCKS[32];
__constant__ int D_F_MAIN_NBLOCKS,D_F_BLOCK_NBLOCKS;

__host__ __device__ static inline MateValue mget(MateID m,int k){return MateValue((m>>(2*k))&3ULL);}
__host__ __device__ static inline MateValuePair mpair(MateID m,int p){return MateValuePair((m>>(2*(p-1)))&15ULL);}
__host__ __device__ static inline MateID mset(MateID m,int k,MateValue v){MateID z=3ULL<<(2*k);return (m&~z)|(MateID(v)<<(2*k));}
__host__ __device__ static inline MateID msetpair(MateID m,int p,MateValuePair v){MateID z=15ULL<<(2*(p-1));return (m&~z)|(MateID(v)<<(2*(p-1)));}
__host__ __device__ static inline MateID mshrink(MateID m,int k){MateID mask=(1ULL<<(2*k))-1ULL;return((m&~mask)>>2)|(m&mask);}
__host__ __device__ static inline MateID minsert(MateID m,int k,MateValue v){MateID lowmask=k?((1ULL<<(2*k))-1ULL):0ULL;MateID lo=m&lowmask,hi=m&~lowmask;return lo|(MateID(v)<<(2*k))|(hi<<2);}

static void ck(cudaError_t e,const char* w){if(e!=cudaSuccess){std::cerr<<w<<": "<<cudaGetErrorString(e)<<"\n";std::exit(1);}}
static void build_full_dp(){for(int h=0;h<=MAXW+1;++h)H_DP[0][h]=(h==0);for(int w=1;w<=MAXW;++w)for(int h=0;h<=MAXW;++h){Code x=H_DP[w-1][h];if(h>0)x+=H_DP[w-1][h-1];if(h<MAXW+1)x+=H_DP[w-1][h+1];H_DP[w][h]=x;}}
static void build_bounded_dp(int cap,Code out[MAXW+1][MAXW+2]){
    std::memset(out,0,sizeof(Code)*(MAXW+1)*(MAXW+2));
    out[0][0]=1;
    for(int w=1;w<=MAXW;++w)for(int h=0;h<=cap;++h){Code x=out[w-1][h];if(h>0)x+=out[w-1][h-1];if(h<cap)x+=out[w-1][h+1];out[w][h]=x;}
}


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


static constexpr uint32_t POW3_7=2187u;
static uint32_t pow3_host(int n){uint32_t z=1;while(n--)z*=3u;return z;}
static uint32_t ternary_index_host(uint32_t code,int len){uint32_t z=0,m=1;for(int p=0;p<len;++p){uint32_t v=(code>>(2*p))&3u;if(v>2){std::cerr<<"bad ternary symbol\n";std::exit(31);}z+=v*m;m*=3u;}return z;}

struct FactorTablesHost {
    static constexpr int STRIDE=MAXW+2;
    std::vector<uint32_t> low_all_codes,low_mask_codes,low_mask_off,low_packed_rank;
    std::vector<uint32_t> high_all_codes,high_mask_codes,high_mask_off,high_packed_rank;
    std::vector<Code> high_main_base,high_block_base;
    std::array<uint32_t,MAXW+2> low_all_off{},high_all_off{};
};
static uint32_t seg_occ(uint32_t code,int len){uint32_t z=0;for(int p=0;p<len;++p)if((code>>(2*p))&3u)z|=1u<<p;return z;}
static FactorTablesHost build_factor_tables(){
    FactorTablesHost f;
    constexpr int L=LOW_LUT_K,H=HIGH_LUT_K,S=FactorTablesHost::STRIDE;
    static_assert(L>0&&H>0,"factorized forced2 requires nonzero LOW/HIGH");
    const uint32_t LM=1u<<L, HM=1u<<H;
    std::vector<std::vector<uint32_t>> la(MAXW+1),ha(MAXW+1);
    std::vector<std::vector<uint32_t>> lm(size_t(LM)*S),hm(size_t(HM)*S);
    // LOW suffixes: start at h, consume positions L-1..0, end at 0.
    for(int h0=0;h0<=L+1;++h0){
        std::function<void(int,int,uint32_t)> rec=[&](int pos,int h,uint32_t code){
            if(pos<0){if(h==0){la[h0].push_back(code);lm[size_t(seg_occ(code,L))*S+h0].push_back(code);}return;}
            if(h<0||h>pos+1)return; // cannot return to zero in remaining positions
            rec(pos-1,h,code);
            if(h>0)rec(pos-1,h-1,code|(uint32_t(R)<<(2*pos)));
            rec(pos-1,h+1,code|(uint32_t(::L)<<(2*pos)));
        };rec(L-1,h0,0);
    }
    // HIGH prefixes: start at height 1, consume high-to-low, group by ending height.
    std::function<void(int,int,uint32_t)> rech=[&](int pos,int h,uint32_t code){
        if(pos<0){ha[h].push_back(code);hm[size_t(seg_occ(code,H))*S+h].push_back(code);return;}
        rech(pos-1,h,code);
        if(h>0)rech(pos-1,h-1,code|(uint32_t(R)<<(2*pos)));
        rech(pos-1,h+1,code|(uint32_t(::L)<<(2*pos)));
    };rech(H-1,1,0);
    f.low_packed_rank.assign(pow3_host(L),0xffffffffu);
    f.high_packed_rank.assign(size_t(1)<<(2*H),0xffffffffu);
    for(int h=0;h<=MAXW;++h){f.low_all_off[h]=f.low_all_codes.size();for(uint32_t r=0;r<la[h].size();++r){auto c=la[h][r];f.low_all_codes.push_back(c);f.low_packed_rank[ternary_index_host(c,L)]=r<<L;} }
    f.low_all_off[MAXW+1]=f.low_all_codes.size();
    f.low_mask_off.resize(size_t(LM)*S);
    for(uint32_t m=0;m<LM;++m)for(int h=0;h<S;++h){size_t ix=size_t(m)*S+h;f.low_mask_off[ix]=f.low_mask_codes.size();if(h<=MAXW)for(uint32_t r=0;r<lm[ix].size();++r){auto c=lm[ix][r];f.low_mask_codes.push_back(c);auto ti=ternary_index_host(c,L);f.low_packed_rank[ti]=(f.low_packed_rank[ti]&~((1u<<L)-1u))|r;}}
    for(int h=0;h<=MAXW;++h){f.high_all_off[h]=f.high_all_codes.size();for(uint32_t r=0;r<ha[h].size();++r){auto c=ha[h][r];f.high_all_codes.push_back(c);f.high_packed_rank[c]=r<<H;}}
    f.high_all_off[MAXW+1]=f.high_all_codes.size();
    f.high_mask_off.resize(size_t(HM)*S);
    for(uint32_t m=0;m<HM;++m)for(int h=0;h<S;++h){size_t ix=size_t(m)*S+h;f.high_mask_off[ix]=f.high_mask_codes.size();if(h<=MAXW)for(uint32_t r=0;r<hm[ix].size();++r){auto c=hm[ix][r];f.high_mask_codes.push_back(c);f.high_packed_rank[c]=(f.high_packed_rank[c]&~((1u<<H)-1u))|r;}}
    auto prefix_base=[&](uint32_t code,int offset){Code rank=0;int h=1;for(int p=H-1;p>=0;--p){MateValue v=MateValue((code>>(2*p))&3u);int fp=offset+p;if(v>N)rank+=H_DP[fp][h];if(v>R&&h>0)rank+=H_DP[fp][h-1];if(v==R)--h;else if(v==::L)++h;}return rank;};
    f.high_main_base.resize(f.high_all_codes.size());f.high_block_base.resize(f.high_all_codes.size());
    for(size_t i=0;i<f.high_all_codes.size();++i){f.high_main_base[i]=prefix_base(f.high_all_codes[i],L+1);f.high_block_base[i]=prefix_base(f.high_all_codes[i],L);}
    std::cerr<<"factor tables low_all="<<f.low_all_codes.size()<<" low_mask="<<f.low_mask_codes.size()<<" low_dense_mib="<<double(f.low_packed_rank.size()*4)/(1<<20)
             <<" high_all="<<f.high_all_codes.size()<<" high_mask="<<f.high_mask_codes.size()<<" high_dense_mib="<<double(f.high_packed_rank.size()*4)/(1<<20)<<"\n";
    return f;
}
static FactorTablesHost G_FACTOR;
static uint32_t factor_count(const std::vector<uint32_t>&off,uint32_t mask,int h){constexpr int S=FactorTablesHost::STRIDE;size_t i=size_t(mask)*S+h;return off[i+1]-off[i];}
static std::vector<FBlock> make_factor_main_blocks(bool fix_low,uint32_t mask){
    constexpr int L=LOW_LUT_K,H=HIGH_LUT_K;std::vector<FBlock> v;v.reserve(3*(H+2));Code off=0;
    for(int he=0;he<=H+1;++he)for(int cv=0;cv<3;++cv){int hs=he+(cv==int(::L)?1:cv==int(::R)?-1:0);uint32_t hc=0,lc=0;if(hs>=0&&hs<=L+1){hc=fix_low?(G_FACTOR.high_all_off[he+1]-G_FACTOR.high_all_off[he]):factor_count(G_FACTOR.high_mask_off,mask,he);lc=fix_low?factor_count(G_FACTOR.low_mask_off,mask,hs):(G_FACTOR.low_all_off[hs+1]-G_FACTOR.low_all_off[hs]);}uint64_t n=uint64_t(hc)*lc;v.push_back({off,off+n,lc,(uint8_t)he,(uint8_t)std::max(0,hs),(uint8_t)cv,0});off+=n;}
    return v;
}
static std::vector<FBlock> make_factor_block_blocks(bool fix_low,uint32_t mask){
    constexpr int H=HIGH_LUT_K;std::vector<FBlock> v;v.reserve(H+2);Code off=0;
    for(int h=0;h<=H+1;++h){uint32_t hc=fix_low?(G_FACTOR.high_all_off[h+1]-G_FACTOR.high_all_off[h]):factor_count(G_FACTOR.high_mask_off,mask,h);uint32_t lc=fix_low?factor_count(G_FACTOR.low_mask_off,mask,h):(G_FACTOR.low_all_off[h+1]-G_FACTOR.low_all_off[h]);uint64_t n=uint64_t(hc)*lc;v.push_back({off,off+n,lc,(uint8_t)h,(uint8_t)h,0,0});off+=n;}return v;
}
struct Interval{Code global,local,len;};
struct HighTablesHost{std::vector<HighEntry> entries;std::vector<uint32_t> offsets;};
static void high_enum_rec(int pos,int h,uint32_t occ,uint32_t code,int lowlen,uint64_t&base,std::vector<HighEntry>&entries){
    if(pos<0){Code cnt=H_DP[lowlen][h];if(!cnt)return;if(base>0xffffffffULL||base+cnt>0x100000000ULL){std::cerr<<"high base overflow\n";std::exit(7);}entries.push_back({code,(uint32_t)base});base+=cnt;return;}
    bool o=(occ>>pos)&1u;
    if(!o){high_enum_rec(pos-1,h,occ,code,lowlen,base,entries);return;}
    if(h>0)high_enum_rec(pos-1,h-1,occ,code|(uint32_t(R)<<(2*pos)),lowlen,base,entries);
    high_enum_rec(pos-1,h+1,occ,code|(uint32_t(::L)<<(2*pos)),lowlen,base,entries);
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
__device__ __forceinline__ Code rank_known_t(Code src_rank,MateID src,MateID dst,int hi,int lo,uint32_t fixed,uint32_t occ,const Code dp[MAXW+1][MAXW+2]){
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

__device__ __forceinline__ MateID bounded_unrank(Code rank,int width,const Code dp[MAXW+1][MAXW+2],int cap){
    MateID m=0;int h=1;
    for(int pos=width-1;pos>=0;--pos){Code z=dp[pos][h];if(rank<z)continue;rank-=z;if(h>0){z=dp[pos][h-1];if(rank<z){m|=MateID(R)<<(2*pos);--h;continue;}rank-=z;}if(h<cap){m|=MateID(L)<<(2*pos);++h;}}
    return m;
}
__device__ __forceinline__ Code bounded_rank(MateID m,int width,const Code dp[MAXW+1][MAXW+2],int cap){
    Code rank=0;int h=1;for(int pos=width-1;pos>=0;--pos){auto v=mget(m,pos);if(v>N)rank+=dp[pos][h];if(v>R&&h>0)rank+=dp[pos][h-1];if(v==R)--h;else if(v==L)++h;}return rank;
}
__global__ void bounded_fill_one_kernel(Count* a,Code n){for(Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,st=Code(gridDim.x)*blockDim.x;i<n;i+=st)a[i]=1;}
__global__ void bounded_fill_row2_automaton_kernel(Count* out,Code n,int width){
    for(Code brank=Code(blockIdx.x)*blockDim.x+threadIdx.x,st=Code(gridDim.x)*blockDim.x;brank<n;brank+=st){
        Code r=brank; int h=1;
        long long v0=1,v1=0,v2=0,v3=0,v4=0,v5=0;
#pragma unroll
        for(int pos=TARGET_W-1;pos>=0;--pos){
            MateValue z=N; Code a=D_BOUND_DP[pos][h];
            if(r<a) z=N;
            else { r-=a; if(h>0){ a=D_BOUND_DP[pos][h-1]; if(r<a) z=R; else {r-=a; z=L;} } else z=L; }
            long long w0=0,w1=0,w2=0,w3=0,w4=0,w5=0;
            if(z==N){w1=v0+2*v1;w2=-v4-v5;w3=v3;w4=-v5;w5=v2+2*v4+3*v5;}
            else if(z==R){w0=v3;w2=v0;w4=v1;--h;}
            else {w0=v5;w1=v2+2*v4+2*v5;w3=v0+v1;++h;}
            v0=w0;v1=w1;v2=w2;v3=w3;v4=w4;v5=w5;
        }
        long long raw=v2+2*v4+2*v5;
        long long q=raw%(long long)D_MOD; if(q<0)q+=D_MOD;
        out[brank]=Count(q);
    }
}
__device__ __forceinline__ Count r3_add_mod(Count a,Count b){Count m=D_MOD;return a>=m-b?a-(m-b):a+b;}
__device__ __forceinline__ Count r3_mul_mod(Count a,Count b){
    unsigned long long z=(unsigned long long)a*b,m=D_MOD,d=(1ULL<<32)-m;
    if(d<65536ULL){unsigned long long t=(uint32_t)z+(z>>32)*d;t=(uint32_t)t+(t>>32)*d;if(t>=m)t-=m;if(t>=m)t-=m;return (Count)t;}
    return (Count)(z%m);
}
__global__ void bounded_fill_row3_lut_kernel(Count*out,Code n,int width){
    constexpr int DIM=oneesan::row3auto::DIM; constexpr int LO=TARGET_W/2;
    for(Code brank=Code(blockIdx.x)*blockDim.x+threadIdx.x,st=Code(gridDim.x)*blockDim.x;brank<n;brank+=st){
        Code r=brank;int h=1;uint32_t ch=0,cl=0;
#pragma unroll
        for(int pos=TARGET_W-1;pos>=0;--pos){int dig=0;Code a=D_BOUND_DP[pos][h];if(r<a)dig=0;else{r-=a;if(h>0){a=D_BOUND_DP[pos][h-1];if(r<a){dig=1;--h;}else{r-=a;dig=2;++h;}}else{dig=2;++h;}}if(pos>=LO)ch=ch*3u+dig;else cl=cl*3u+dig;}
        const Count*pv=D_R3_PREF+(size_t)ch*DIM;const Count*sv=D_R3_SUFF+(size_t)cl*DIM;Count acc=0;
#pragma unroll
        for(int i=0;i<DIM;++i)acc=r3_add_mod(acc,r3_mul_mod(pv[i],sv[i]));
        out[brank]=r3_mul_mod(acc,D_R3_INV_SCALE);
    }
}

__device__ __forceinline__ int r4_block_size(int h){return h==0?19:h==1?20:h==2?12:h==3?4:h==4?1:0;}
__global__ void bounded_fill_row4_lut_kernel(Count*out,Code n,int width){
    constexpr int STRIDE=20,LO=TARGET_W/2;
    for(Code brank=Code(blockIdx.x)*blockDim.x+threadIdx.x,st=Code(gridDim.x)*blockDim.x;brank<n;brank+=st){
        Code r=brank;int h=1,hs=0;uint32_t ch=0,cl=0;
#pragma unroll
        for(int pos=TARGET_W-1;pos>=0;--pos){int dig=0;Code a=D_BOUND_DP[pos][h];if(r<a)dig=0;else{r-=a;if(h>0){a=D_BOUND_DP[pos][h-1];if(r<a){dig=1;--h;}else{r-=a;dig=2;++h;}}else{dig=2;++h;}}if(pos>=LO){ch=ch*3u+dig;if(pos==LO)hs=h;}else cl=cl*3u+dig;}
        const Count*pv=D_R4_PREF+(size_t)ch*STRIDE;const Count*sv=D_R4_SUFF+(size_t)cl*STRIDE;Count acc=0;int dim=r4_block_size(hs);
#pragma unroll 4
        for(int i=0;i<dim;++i)acc=r3_add_mod(acc,r3_mul_mod(pv[i],sv[i]));out[brank]=acc;
    }
}

__device__ __forceinline__ int r5_block_size(int h){return h==0?51:h==1?61:h==2?40:h==3?18:h==4?5:h==5?1:0;}
__global__ void bounded_fill_row5_full_kernel(Count*fullMain,Code n,int width){
    constexpr int STRIDE=61,LO=TARGET_W/2;
    for(Code brank=Code(blockIdx.x)*blockDim.x+threadIdx.x,st=Code(gridDim.x)*blockDim.x;brank<n;brank+=st){
        Code r=brank,grank=0;int h=1,hs=0;uint32_t ch=0,cl=0;
#pragma unroll
        for(int pos=TARGET_W-1;pos>=0;--pos){
            int dig=0;Code a=D_BOUND_DP[pos][h];
            if(r<a)dig=0;
            else{r-=a;if(h>0){a=D_BOUND_DP[pos][h-1];if(r<a){dig=1;grank+=D_FULL_DP[pos][h];--h;}else{r-=a;dig=2;grank+=D_FULL_DP[pos][h]+D_FULL_DP[pos][h-1];++h;}}else{dig=2;grank+=D_FULL_DP[pos][h];++h;}}
            if(pos>=LO){ch=ch*3u+dig;if(pos==LO)hs=h;}else cl=cl*3u+dig;
        }
        const Count*pv=D_R5_PREF+D_R5_PREF_IDX[ch];const Count*sv=D_R5_SUFF+D_R5_SUFF_IDX[cl];Count acc=0;int dim=r5_block_size(hs);
#pragma unroll 4
        for(int i=0;i<dim;++i)acc=r3_add_mod(acc,r3_mul_mod(pv[i],sv[i]));
        fullMain[grank]=acc;
    }
}

__global__ void bounded_fill_row5_full_warp_kernel(Count*fullMain,Code n,int width){
    constexpr int STRIDE=61,LO=TARGET_W/2;int lane=threadIdx.x&31;Code warp=(Code(blockIdx.x)*blockDim.x+threadIdx.x)>>5,step=(Code(gridDim.x)*blockDim.x)>>5;
    for(Code brank=warp;brank<n;brank+=step){
        unsigned long long meta=0;Code grank=0;
        if(lane==0){Code r=brank;int h=1,hs=0;uint32_t ch=0,cl=0;
#pragma unroll
            for(int pos=TARGET_W-1;pos>=0;--pos){int dig=0;Code a=D_BOUND_DP[pos][h];if(r<a)dig=0;else{r-=a;if(h>0){a=D_BOUND_DP[pos][h-1];if(r<a){dig=1;grank+=D_FULL_DP[pos][h];--h;}else{r-=a;dig=2;grank+=D_FULL_DP[pos][h]+D_FULL_DP[pos][h-1];++h;}}else{dig=2;grank+=D_FULL_DP[pos][h];++h;}}if(pos>=LO){ch=ch*3u+dig;if(pos==LO)hs=h;}else cl=cl*3u+dig;}meta=uint64_t(ch)|(uint64_t(cl)<<24)|(uint64_t(hs)<<48);}
        meta=__shfl_sync(0xffffffffu,meta,0);grank=__shfl_sync(0xffffffffu,grank,0);uint32_t ch=uint32_t(meta&0xffffffu),cl=uint32_t((meta>>24)&0xffffffu);int hs=int((meta>>48)&0xffu),dim=r5_block_size(hs);const Count*pv=D_R5_PREF+D_R5_PREF_IDX[ch];const Count*sv=D_R5_SUFF+D_R5_SUFF_IDX[cl];Count acc=0;for(int i=lane;i<dim;i+=32)acc=r3_add_mod(acc,r3_mul_mod(pv[i],sv[i]));
        for(int off=16;off;off>>=1)acc=r3_add_mod(acc,__shfl_down_sync(0xffffffffu,acc,off));if(lane==0)fullMain[grank]=acc;
    }
}

__global__ void bounded_fill_row5_full_lane8_kernel(Count*fullMain,Code n,int width){
    constexpr int STRIDE=61,LO=TARGET_W/2;int lane=threadIdx.x&7,warpLane=threadIdx.x&31,leader=warpLane&~7;unsigned mask=0xffu<<leader;Code group=(Code(blockIdx.x)*blockDim.x+threadIdx.x)>>3,step=(Code(gridDim.x)*blockDim.x)>>3;
    for(Code brank=group;brank<n;brank+=step){
        unsigned long long meta=0;Code grank=0;if(lane==0){Code r=brank;int h=1,hs=0;uint32_t ch=0,cl=0;
#pragma unroll
            for(int pos=TARGET_W-1;pos>=0;--pos){int dig=0;Code a=D_BOUND_DP[pos][h];if(r<a)dig=0;else{r-=a;if(h>0){a=D_BOUND_DP[pos][h-1];if(r<a){dig=1;grank+=D_FULL_DP[pos][h];--h;}else{r-=a;dig=2;grank+=D_FULL_DP[pos][h]+D_FULL_DP[pos][h-1];++h;}}else{dig=2;grank+=D_FULL_DP[pos][h];++h;}}if(pos>=LO){ch=ch*3u+dig;if(pos==LO)hs=h;}else cl=cl*3u+dig;}meta=uint64_t(ch)|(uint64_t(cl)<<24)|(uint64_t(hs)<<48);}
        meta=__shfl_sync(mask,meta,leader);grank=__shfl_sync(mask,grank,leader);uint32_t ch=uint32_t(meta&0xffffffu),cl=uint32_t((meta>>24)&0xffffffu);int hs=int((meta>>48)&0xffu),dim=r5_block_size(hs);const Count*pv=D_R5_PREF+D_R5_PREF_IDX[ch];const Count*sv=D_R5_SUFF+D_R5_SUFF_IDX[cl];Count acc=0;for(int i=lane;i<dim;i+=8)acc=r3_add_mod(acc,r3_mul_mod(pv[i],sv[i]));
        for(int off=4;off;off>>=1)acc=r3_add_mod(acc,__shfl_down_sync(mask,acc,off,8));if(lane==0)fullMain[grank]=acc;
    }
}

template<int LANES> __global__ void bounded_fill_row5_full_group_kernel(Count*fullMain,Code n,int width){
    static_assert(LANES==4||LANES==8||LANES==16);constexpr int STRIDE=61,LO=TARGET_W/2;int lane=threadIdx.x&(LANES-1),warpLane=threadIdx.x&31,leader=warpLane&~(LANES-1);unsigned mask=((1u<<LANES)-1u)<<leader;Code group=(Code(blockIdx.x)*blockDim.x+threadIdx.x)/LANES,step=(Code(gridDim.x)*blockDim.x)/LANES;
    for(Code brank=group;brank<n;brank+=step){unsigned long long meta=0;Code grank=0;if(lane==0){Code r=brank;int h=1,hs=0;uint32_t ch=0,cl=0;
#pragma unroll
        for(int pos=TARGET_W-1;pos>=0;--pos){int dig=0;Code a=D_BOUND_DP[pos][h];if(r<a)dig=0;else{r-=a;if(h>0){a=D_BOUND_DP[pos][h-1];if(r<a){dig=1;grank+=D_FULL_DP[pos][h];--h;}else{r-=a;dig=2;grank+=D_FULL_DP[pos][h]+D_FULL_DP[pos][h-1];++h;}}else{dig=2;grank+=D_FULL_DP[pos][h];++h;}}if(pos>=LO){ch=ch*3u+dig;if(pos==LO)hs=h;}else cl=cl*3u+dig;}meta=uint64_t(ch)|(uint64_t(cl)<<24)|(uint64_t(hs)<<48);}
        meta=__shfl_sync(mask,meta,leader);grank=__shfl_sync(mask,grank,leader);uint32_t ch=uint32_t(meta&0xffffffu),cl=uint32_t((meta>>24)&0xffffffu);int hs=int((meta>>48)&0xffu),dim=r5_block_size(hs);const Count*pv=D_R5_PREF+D_R5_PREF_IDX[ch];const Count*sv=D_R5_SUFF+D_R5_SUFF_IDX[cl];Count acc=0;for(int i=lane;i<dim;i+=LANES)acc=r3_add_mod(acc,r3_mul_mod(pv[i],sv[i]));for(int off=LANES/2;off;off>>=1)acc=r3_add_mod(acc,__shfl_down_sync(mask,acc,off,LANES));if(lane==0)fullMain[grank]=acc;
    }
}


__global__ void bounded_fill_row5_full_range_kernel(Count*fullMain,Code blo,Code bhi,Code fullBase){
    constexpr int STRIDE=61,LO=TARGET_W/2;
    for(Code brank=blo+Code(blockIdx.x)*blockDim.x+threadIdx.x,st=Code(gridDim.x)*blockDim.x;brank<bhi;brank+=st){Code r=brank,grank=0;int h=1,hs=0;uint32_t ch=0,cl=0;
#pragma unroll
        for(int pos=TARGET_W-1;pos>=0;--pos){int dig=0;Code a=D_BOUND_DP[pos][h];if(r<a)dig=0;else{r-=a;if(h>0){a=D_BOUND_DP[pos][h-1];if(r<a){dig=1;grank+=D_FULL_DP[pos][h];--h;}else{r-=a;dig=2;grank+=D_FULL_DP[pos][h]+D_FULL_DP[pos][h-1];++h;}}else{dig=2;grank+=D_FULL_DP[pos][h];++h;}}if(pos>=LO){ch=ch*3u+dig;if(pos==LO)hs=h;}else cl=cl*3u+dig;}
        const Count*pv=D_R5_PREF+D_R5_PREF_IDX[ch];const Count*sv=D_R5_SUFF+D_R5_SUFF_IDX[cl];Count acc=0,dim=r5_block_size(hs);for(int i=0;i<dim;++i)acc=r3_add_mod(acc,r3_mul_mod(pv[i],sv[i]));fullMain[grank-fullBase]=acc;}
}
template<int LANES> __global__ void bounded_fill_row5_full_group_range_kernel(Count*fullMain,Code blo,Code bhi,Code fullBase){
    static_assert(LANES==4||LANES==8||LANES==16);constexpr int STRIDE=61,LO=TARGET_W/2;int lane=threadIdx.x&(LANES-1),warpLane=threadIdx.x&31,leader=warpLane&~(LANES-1);unsigned mask=((1u<<LANES)-1u)<<leader;Code group=(Code(blockIdx.x)*blockDim.x+threadIdx.x)/LANES,step=(Code(gridDim.x)*blockDim.x)/LANES;
    for(Code brank=blo+group;brank<bhi;brank+=step){unsigned long long meta=0;Code grank=0;if(lane==0){Code r=brank;int h=1,hs=0;uint32_t ch=0,cl=0;
#pragma unroll
        for(int pos=TARGET_W-1;pos>=0;--pos){int dig=0;Code a=D_BOUND_DP[pos][h];if(r<a)dig=0;else{r-=a;if(h>0){a=D_BOUND_DP[pos][h-1];if(r<a){dig=1;grank+=D_FULL_DP[pos][h];--h;}else{r-=a;dig=2;grank+=D_FULL_DP[pos][h]+D_FULL_DP[pos][h-1];++h;}}else{dig=2;grank+=D_FULL_DP[pos][h];++h;}}if(pos>=LO){ch=ch*3u+dig;if(pos==LO)hs=h;}else cl=cl*3u+dig;}meta=uint64_t(ch)|(uint64_t(cl)<<24)|(uint64_t(hs)<<48);}
        meta=__shfl_sync(mask,meta,leader);grank=__shfl_sync(mask,grank,leader);uint32_t ch=uint32_t(meta&0xffffffu),cl=uint32_t((meta>>24)&0xffffffu);int hs=int((meta>>48)&0xffu),dim=r5_block_size(hs);const Count*pv=D_R5_PREF+D_R5_PREF_IDX[ch];const Count*sv=D_R5_SUFF+D_R5_SUFF_IDX[cl];Count acc=0;for(int i=lane;i<dim;i+=LANES)acc=r3_add_mod(acc,r3_mul_mod(pv[i],sv[i]));for(int off=LANES/2;off;off>>=1)acc=r3_add_mod(acc,__shfl_down_sync(mask,acc,off,LANES));if(lane==0)fullMain[grank-fullBase]=acc;}
}


__host__ __device__ __forceinline__ int r6_block_size(int h){return h==0?141:h==1?182:h==2?135:h==3?68:h==4?25:h==5?6:h==6?1:0;}
__device__ __forceinline__ int r6_gpu_delta(int sym){return sym==0?0:(sym==1?-1:1);}
__global__ void r6_prefix_level_kernel(Code parents,const Count*cur,const int8_t*hp,Count*nxt,int8_t*hn,const Count*mat){
    Code children=parents*3;for(Code q=Code(blockIdx.x)*blockDim.x+threadIdx.x,st=Code(gridDim.x)*blockDim.x;q<children;q+=st){Code p=q/3;int sym=int(q-p*3),h=hp[p];if(h<0){hn[q]=-1;continue;}int h2=h+r6_gpu_delta(sym);if(h2<0||h2>6){hn[q]=-1;continue;}hn[q]=int8_t(h2);int ns=r6_block_size(h),nd=r6_block_size(h2);const Count*M=mat+D_R6_GPU_MAT_OFF[sym][h];for(int d=0;d<nd;++d){Count acc=0;for(int a=0;a<ns;++a){Count u=cur[Code(a)*parents+p],c=M[a*nd+d];if(u&&c)acc=r3_add_mod(acc,r3_mul_mod(u,c));}nxt[Code(d)*children+q]=acc;}}
}
__global__ void r6_suffix_level_kernel(Code parents,const Count*cur,const int8_t*hp,Count*nxt,int8_t*hn,const Count*mat){
    Code children=parents*3;for(Code q=Code(blockIdx.x)*blockDim.x+threadIdx.x,st=Code(gridDim.x)*blockDim.x;q<children;q+=st){int sym=int(q/parents);Code p=q-Code(sym)*parents;int h=hp[p];if(h<0){hn[q]=-1;continue;}int hs=h-r6_gpu_delta(sym);if(hs<0||hs>6){hn[q]=-1;continue;}hn[q]=int8_t(hs);int ns=r6_block_size(hs),nd=r6_block_size(h);const Count*M=mat+D_R6_GPU_MAT_OFF[sym][hs];for(int a=0;a<ns;++a){Count acc=0;const Count*row=M+a*nd;for(int d=0;d<nd;++d){Count u=cur[Code(d)*parents+p],c=row[d];if(u&&c)acc=r3_add_mod(acc,r3_mul_mod(c,u));}nxt[Code(a)*children+q]=acc;}}
}
__global__ void r6_pack_level_kernel(Code codes,const Count*dense,const int8_t*height,const uint32_t*idx,Count*packed){for(Code q=Code(blockIdx.x)*blockDim.x+threadIdx.x,st=Code(gridDim.x)*blockDim.x;q<codes;q+=st){int h=height[q];if(h<0)continue;uint32_t o=idx[q];int n=r6_block_size(h);for(int j=0;j<n;++j)packed[Code(o)+j]=dense[Code(j)*codes+q];}}
template<int LANES> __global__ void bounded_fill_row6_dense_group_range_kernel(Count*fullMain,Code blo,Code bhi,Code fullBase,const Count*pref,Code pn,const Count*suff,Code sn){
    static_assert(LANES==4||LANES==8||LANES==16);constexpr int LO=TARGET_W/2;int lane=threadIdx.x&(LANES-1),warpLane=threadIdx.x&31,leader=warpLane&~(LANES-1);unsigned mask=((1u<<LANES)-1u)<<leader;Code group=(Code(blockIdx.x)*blockDim.x+threadIdx.x)/LANES,step=(Code(gridDim.x)*blockDim.x)/LANES;
    for(Code brank=blo+group;brank<bhi;brank+=step){unsigned long long meta=0;Code grank=0;if(lane==0){Code r=brank;int h=1,hs=0;uint32_t ch=0,cl=0;
        for(int pos=TARGET_W-1;pos>=0;--pos){int dig=0;Code a=D_BOUND_DP[pos][h];if(r<a)dig=0;else{r-=a;if(h>0){a=D_BOUND_DP[pos][h-1];if(r<a){dig=1;grank+=D_FULL_DP[pos][h];--h;}else{r-=a;dig=2;grank+=D_FULL_DP[pos][h]+D_FULL_DP[pos][h-1];++h;}}else{dig=2;grank+=D_FULL_DP[pos][h];++h;}}if(pos>=LO){ch=ch*3u+dig;if(pos==LO)hs=h;}else cl=cl*3u+dig;}meta=uint64_t(ch)|(uint64_t(cl)<<24)|(uint64_t(hs)<<48);}
        meta=__shfl_sync(mask,meta,leader);grank=__shfl_sync(mask,grank,leader);uint32_t ch=uint32_t(meta&0xffffffu),cl=uint32_t((meta>>24)&0xffffffu);int hs=int((meta>>48)&0xffu),dim=r6_block_size(hs);Count acc=0;for(int j=lane;j<dim;j+=LANES){Count a=pref[Code(j)*pn+ch],b=suff[Code(j)*sn+cl];if(a&&b)acc=r3_add_mod(acc,r3_mul_mod(a,b));}for(int off=LANES/2;off;off>>=1)acc=r3_add_mod(acc,__shfl_down_sync(mask,acc,off,LANES));if(lane==0)fullMain[grank-fullBase]=acc;}
}
__global__ void bounded_fill_row6_dense_range_kernel(Count*fullMain,Code blo,Code bhi,Code fullBase,const Count*pref,Code pn,const Count*suff,Code sn){constexpr int LO=TARGET_W/2;for(Code brank=blo+Code(blockIdx.x)*blockDim.x+threadIdx.x,st=Code(gridDim.x)*blockDim.x;brank<bhi;brank+=st){Code r=brank,grank=0;int h=1,hs=0;uint32_t ch=0,cl=0;for(int pos=TARGET_W-1;pos>=0;--pos){int dig=0;Code a=D_BOUND_DP[pos][h];if(r<a)dig=0;else{r-=a;if(h>0){a=D_BOUND_DP[pos][h-1];if(r<a){dig=1;grank+=D_FULL_DP[pos][h];--h;}else{r-=a;dig=2;grank+=D_FULL_DP[pos][h]+D_FULL_DP[pos][h-1];++h;}}else{dig=2;grank+=D_FULL_DP[pos][h];++h;}}if(pos>=LO){ch=ch*3u+dig;if(pos==LO)hs=h;}else cl=cl*3u+dig;}Count acc=0,dim=r6_block_size(hs);for(int j=0;j<dim;++j){Count a=pref[Code(j)*pn+ch],b=suff[Code(j)*sn+cl];if(a&&b)acc=r3_add_mod(acc,r3_mul_mod(a,b));}fullMain[grank-fullBase]=acc;}}
__global__ void bounded_fill_row6_full_range_kernel(Count*fullMain,Code blo,Code bhi,Code fullBase){
    constexpr int LO=TARGET_W/2;
    for(Code brank=blo+Code(blockIdx.x)*blockDim.x+threadIdx.x,st=Code(gridDim.x)*blockDim.x;brank<bhi;brank+=st){Code r=brank,grank=0;int h=1,hs=0;uint32_t ch=0,cl=0;
#pragma unroll
        for(int pos=TARGET_W-1;pos>=0;--pos){int dig=0;Code a=D_BOUND_DP[pos][h];if(r<a)dig=0;else{r-=a;if(h>0){a=D_BOUND_DP[pos][h-1];if(r<a){dig=1;grank+=D_FULL_DP[pos][h];--h;}else{r-=a;dig=2;grank+=D_FULL_DP[pos][h]+D_FULL_DP[pos][h-1];++h;}}else{dig=2;grank+=D_FULL_DP[pos][h];++h;}}if(pos>=LO){ch=ch*3u+dig;if(pos==LO)hs=h;}else cl=cl*3u+dig;}
        const Count*pv=D_R6_PREF+D_R6_PREF_IDX[ch];const Count*sv=D_R6_SUFF+D_R6_SUFF_IDX[cl];Count acc=0;int dim=r6_block_size(hs);for(int i=0;i<dim;++i)acc=r3_add_mod(acc,r3_mul_mod(pv[i],sv[i]));fullMain[grank-fullBase]=acc;}
}
template<int LANES> __global__ void bounded_fill_row6_full_group_range_kernel(Count*fullMain,Code blo,Code bhi,Code fullBase){
    static_assert(LANES==4||LANES==8||LANES==16);constexpr int LO=TARGET_W/2;int lane=threadIdx.x&(LANES-1),warpLane=threadIdx.x&31,leader=warpLane&~(LANES-1);unsigned mask=((1u<<LANES)-1u)<<leader;Code group=(Code(blockIdx.x)*blockDim.x+threadIdx.x)/LANES,step=(Code(gridDim.x)*blockDim.x)/LANES;
    for(Code brank=blo+group;brank<bhi;brank+=step){unsigned long long meta=0;Code grank=0;if(lane==0){Code r=brank;int h=1,hs=0;uint32_t ch=0,cl=0;
#pragma unroll
        for(int pos=TARGET_W-1;pos>=0;--pos){int dig=0;Code a=D_BOUND_DP[pos][h];if(r<a)dig=0;else{r-=a;if(h>0){a=D_BOUND_DP[pos][h-1];if(r<a){dig=1;grank+=D_FULL_DP[pos][h];--h;}else{r-=a;dig=2;grank+=D_FULL_DP[pos][h]+D_FULL_DP[pos][h-1];++h;}}else{dig=2;grank+=D_FULL_DP[pos][h];++h;}}if(pos>=LO){ch=ch*3u+dig;if(pos==LO)hs=h;}else cl=cl*3u+dig;}meta=uint64_t(ch)|(uint64_t(cl)<<24)|(uint64_t(hs)<<48);}
        meta=__shfl_sync(mask,meta,leader);grank=__shfl_sync(mask,grank,leader);uint32_t ch=uint32_t(meta&0xffffffu),cl=uint32_t((meta>>24)&0xffffffu);int hs=int((meta>>48)&0xffu),dim=r6_block_size(hs);const Count*pv=D_R6_PREF+D_R6_PREF_IDX[ch];const Count*sv=D_R6_SUFF+D_R6_SUFF_IDX[cl];Count acc=0;for(int i=lane;i<dim;i+=LANES)acc=r3_add_mod(acc,r3_mul_mod(pv[i],sv[i]));for(int off=LANES/2;off;off>>=1)acc=r3_add_mod(acc,__shfl_down_sync(mask,acc,off,LANES));if(lane==0)fullMain[grank-fullBase]=acc;}
}

__global__ void bounded_embed_kernel(const Count*old,Code oldN,Count*neu,int width){for(Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,st=Code(gridDim.x)*blockDim.x;i<oldN;i+=st){Count c=old[i];if(!c)continue;MateID m=bounded_unrank(i,width,D_BOUND_OLD_DP,D_BOUND_OLD_CAP);Code j=bounded_rank(m,width,D_BOUND_DP,D_BOUND_CAP);neu[j]=c;}}
__global__ void bounded_scatter_full_kernel(const Count*in,Code n,int width){for(Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,st=Code(gridDim.x)*blockDim.x;i<n;i+=st){Count c=in[i];if(!c)continue;MateID m=bounded_unrank(i,width,D_BOUND_DP,D_BOUND_CAP);global_store_main(rank_full_t<TARGET_W>(m),c);}}



__global__ void bounded_inplace_expand_kernel(const Count*tmp,Code lo,Code n,Count*fullMain,int width){
    for(Code q=Code(blockIdx.x)*blockDim.x+threadIdx.x,st=Code(gridDim.x)*blockDim.x;q<n;q+=st){Count c=tmp[q];if(!c)continue;Code br=lo+q;MateID m=bounded_unrank(br,width,D_BOUND_DP,D_BOUND_CAP);Code j=rank_full_t<TARGET_W>(m);fullMain[j]=c;}
}
static void bounded_expand_inplace_single_gpu(Count*fullMain,Code compactN,int W,int threads){
    const Code chunk=Code(1)<<22;Count*tmp=nullptr;ck(cudaMalloc(&tmp,size_t(chunk)*sizeof(Count)),"inplace tmp");auto t0=std::chrono::steady_clock::now();
    for(Code hi=compactN;hi;){Code lo=hi>chunk?hi-chunk:0,n=hi-lo;ck(cudaMemcpy(tmp,fullMain+lo,size_t(n)*sizeof(Count),cudaMemcpyDeviceToDevice),"inplace copy");ck(cudaMemset(fullMain+lo,0,size_t(n)*sizeof(Count)),"inplace clear");int b=int(std::min<Code>(65535,(n+threads-1)/threads));bounded_inplace_expand_kernel<<<std::max(1,b),threads>>>(tmp,lo,n,fullMain,W);ck(cudaDeviceSynchronize(),"inplace expand");hi=lo;}
    double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();cudaFree(tmp);std::cerr<<"bounded inplace expand states="<<compactN<<" sec="<<sec<<"\n";
}

__device__ __forceinline__ int f_find_main(Code i){int lo=0,hi=D_F_MAIN_NBLOCKS;while(lo<hi){int m=(lo+hi)>>1;if(i<D_F_MAIN_BLOCKS[m].end)hi=m;else lo=m+1;}return lo;}
__device__ __forceinline__ int f_find_block(Code i){int lo=0,hi=D_F_BLOCK_NBLOCKS;while(lo<hi){int m=(lo+hi)>>1;if(i<D_F_BLOCK_BLOCKS[m].end)hi=m;else lo=m+1;}return lo;}
__device__ __forceinline__ MateID factor_unrank_main(Code i){
    constexpr int L=LOW_LUT_K,H=HIGH_LUT_K,S=MAXW+2;int b=f_find_main(i);FBlock x=D_F_MAIN_BLOCKS[b];Code r=i-x.off;uint32_t hr=x.stride?uint32_t(r/x.stride):0,lr=x.stride?uint32_t(r-Code(hr)*x.stride):0;uint32_t hc,lc;
    if(D_F_FIX_LOW){hc=D_F_HIGH_ALL_CODES[D_F_HIGH_ALL_OFF[x.he]+hr];lc=D_F_LOW_MASK_CODES[D_F_LOW_MASK_OFF[size_t(D_F_MASK)*S+x.hs]+lr];}
    else{hc=D_F_HIGH_MASK_CODES[D_F_HIGH_MASK_OFF[size_t(D_F_MASK)*S+x.he]+hr];lc=D_F_LOW_ALL_CODES[D_F_LOW_ALL_OFF[x.hs]+lr];}
    return MateID(lc)|(MateID(x.c)<<(2*L))|(MateID(hc)<<(2*(L+1)));
}
__device__ __forceinline__ MateID factor_unrank_block(Code i){
    constexpr int L=LOW_LUT_K,H=HIGH_LUT_K,S=MAXW+2;int b=f_find_block(i);FBlock x=D_F_BLOCK_BLOCKS[b];Code r=i-x.off;uint32_t hr=x.stride?uint32_t(r/x.stride):0,lr=x.stride?uint32_t(r-Code(hr)*x.stride):0;uint32_t hc,lc;
    if(D_F_FIX_LOW){hc=D_F_HIGH_ALL_CODES[D_F_HIGH_ALL_OFF[x.he]+hr];lc=D_F_LOW_MASK_CODES[D_F_LOW_MASK_OFF[size_t(D_F_MASK)*S+x.hs]+lr];}
    else{hc=D_F_HIGH_MASK_CODES[D_F_HIGH_MASK_OFF[size_t(D_F_MASK)*S+x.he]+hr];lc=D_F_LOW_ALL_CODES[D_F_LOW_ALL_OFF[x.hs]+lr];}
    return MateID(lc)|(MateID(hc)<<(2*L));
}
__device__ __forceinline__ int seg_end_height(uint32_t code,int len){constexpr uint64_t E=0x5555555555555555ULL;uint64_t m=(len==16?0xffffffffULL:((1ULL<<(2*len))-1));uint64_t e=E&m;int nr=__popcll(uint64_t(code)&e),nl=__popcll((uint64_t(code)>>1)&e);return 1+nl-nr;}
__device__ __forceinline__ uint32_t ternary_index_dev(uint32_t code,int len){uint32_t lo=__ldg(D_TRIT7_PTR+(code&0x3fffu));if(len<=7)return lo;uint32_t hi=__ldg(D_TRIT7_PTR+((code>>14)&0x3fffu));return lo+POW3_7*hi;}
__device__ __forceinline__ Code factor_rank_main(MateID m){
    constexpr int L=LOW_LUT_K,H=HIGH_LUT_K;constexpr uint32_t LM=(1u<<(2*L))-1u,HM=(1u<<(2*H))-1u;uint32_t lc=uint32_t(m)&LM,hc=uint32_t((m>>(2*(L+1)))&HM);int he=seg_end_height(hc,H);int cv=int(mget(m,L));int bid=3*he+cv;FBlock x=D_F_MAIN_BLOCKS[bid];uint32_t lp=D_F_LOW_PACKED_RANK[ternary_index_dev(lc,L)],hp=D_F_HIGH_PACKED_RANK[hc];uint32_t lr=D_F_FIX_LOW?(lp&((1u<<L)-1u)):(lp>>L);uint32_t hr=D_F_FIX_LOW?(hp>>H):(hp&((1u<<H)-1u));return x.off+Code(hr)*x.stride+lr;
}
__device__ __forceinline__ Code factor_rank_block(MateID m){
    constexpr int L=LOW_LUT_K,H=HIGH_LUT_K;constexpr uint32_t LM=(1u<<(2*L))-1u,HM=(1u<<(2*H))-1u;uint32_t lc=uint32_t(m)&LM,hc=uint32_t((m>>(2*L))&HM);int h=seg_end_height(hc,H);FBlock x=D_F_BLOCK_BLOCKS[h];uint32_t lp=D_F_LOW_PACKED_RANK[ternary_index_dev(lc,L)],hp=D_F_HIGH_PACKED_RANK[hc];uint32_t lr=D_F_FIX_LOW?(lp&((1u<<L)-1u)):(lp>>L);uint32_t hr=D_F_FIX_LOW?(hp>>H):(hp&((1u<<H)-1u));return x.off+Code(hr)*x.stride+lr;
}
__device__ __forceinline__ Code factor_global_rank_main(MateID m){
    constexpr int L=LOW_LUT_K,H=HIGH_LUT_K;constexpr uint32_t LM=(1u<<(2*L))-1u,HM=(1u<<(2*H))-1u;uint32_t lc=uint32_t(m)&LM,hc=uint32_t((m>>(2*(L+1)))&HM);int he=seg_end_height(hc,H);uint32_t hp=D_F_HIGH_PACKED_RANK[hc],lp=D_F_LOW_PACKED_RANK[ternary_index_dev(lc,L)];uint32_t har=hp>>H,lar=lp>>L;Code rank=D_F_HIGH_MAIN_BASE[D_F_HIGH_ALL_OFF[he]+har];MateValue c=mget(m,L);if(c>N)rank+=D_FULL_DP[L][he];if(c>R&&he>0)rank+=D_FULL_DP[L][he-1];return rank+lar;
}
__device__ __forceinline__ Code factor_global_rank_block(MateID m){
    constexpr int L=LOW_LUT_K,H=HIGH_LUT_K;constexpr uint32_t LM=(1u<<(2*L))-1u,HM=(1u<<(2*H))-1u;uint32_t lc=uint32_t(m)&LM,hc=uint32_t((m>>(2*L))&HM);int he=seg_end_height(hc,H);uint32_t hp=D_F_HIGH_PACKED_RANK[hc],lp=D_F_LOW_PACKED_RANK[ternary_index_dev(lc,L)];uint32_t har=hp>>H,lar=lp>>L;return D_F_HIGH_BLOCK_BASE[D_F_HIGH_ALL_OFF[he]+har]+lar;
}
__global__ void init_after_first_row_kernel(int shard,int ng){
    constexpr Code total=Code(1)<<(TARGET_W-1);
    Code tid=Code(blockIdx.x)*blockDim.x+threadIdx.x;
    Code stride=Code(gridDim.x)*blockDim.x;
    for(Code mask=Code(shard)+tid*Code(ng);mask<total;mask+=stride*Code(ng)){
        MateID m=MateID(oneesan::gridfp::R)<<(2*(TARGET_W-1));
#pragma unroll
        for(int p=TARGET_W-1;p>=1;--p){
            if((mask>>(p-1))&1ULL){
                auto z=oneesan::gridfp::include_horizontal(m,TARGET_W,p);
                m=z.mate;
            }
        }
        global_store_main(factor_global_rank_main(m),Count(1));
    }
}

__device__ __forceinline__ void atomic_add_mod(Count*p,Count v);
__global__ void sum_first_row_states_kernel(int shard,int ng,Count* out){
    extern __shared__ unsigned long long sh[];
    constexpr Code total=Code(1)<<(TARGET_W-1);
    Code tid=Code(blockIdx.x)*blockDim.x+threadIdx.x;
    Code stride=Code(gridDim.x)*blockDim.x;
    unsigned long long acc=0;
    for(Code mask=Code(shard)+tid*Code(ng);mask<total;mask+=stride*Code(ng)){
        MateID m=MateID(oneesan::gridfp::R)<<(2*(TARGET_W-1));
#pragma unroll
        for(int p=TARGET_W-1;p>=1;--p){
            if((mask>>(p-1))&1ULL){auto z=oneesan::gridfp::include_horizontal(m,TARGET_W,p);m=z.mate;}
        }
        acc += global_load_main(factor_global_rank_main(m));
    }
    sh[threadIdx.x]=acc; __syncthreads();
    for(unsigned s=blockDim.x/2;s;s>>=1){if(threadIdx.x<s)sh[threadIdx.x]+=sh[threadIdx.x+s];__syncthreads();}
    if(threadIdx.x==0){Count v=Count(sh[0]%D_MOD);atomic_add_mod(out,v);}
}

__global__ void gather_main_kernel(Count*out,MateID*mates,Code n){Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;for(;i<n;i+=stride){MateID m=factor_unrank_main(i);Code g=factor_global_rank_main(m);out[i]=global_load_main(g);if(mates)mates[i]=m;}}
__global__ void gather_block_kernel(Count*out,Code n){Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;for(;i<n;i+=stride){MateID m=factor_unrank_block(i);Code g=factor_global_rank_block(m);out[i]=global_load_block(g);}}
__global__ void scatter_main_kernel(const Count*in,Code n){Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;for(;i<n;i+=stride){MateID m=factor_unrank_main(i);Code g=factor_global_rank_main(m);global_store_main(g,in[i]);}}
__global__ void scatter_block_kernel(const Count*in,Code n){Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;for(;i<n;i+=stride){MateID m=factor_unrank_block(i);Code g=factor_global_rank_block(m);global_store_block(g,in[i]);}}


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

__device__ __forceinline__ void atomic_add_mod(Count*p,Count v){if(!v)return;Count mod=D_MOD;cuda::atomic_ref<Count,cuda::thread_scope_device> a(*p);Count old=a.load(cuda::memory_order_relaxed);for(;;){Count neu=(old>=mod-v)?old-(mod-v):old+v;if(a.compare_exchange_weak(old,neu,cuda::memory_order_relaxed,cuda::memory_order_relaxed))return;}}
__global__ void bounded_main_kernel(const Count*in,Code n,Count*outM,Count*outD,int width,int p){for(Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,st=Code(gridDim.x)*blockDim.x;i<n;i+=st){Count c=in[i];if(!c)continue;MateID m=bounded_unrank(i,width,D_BOUND_DP,D_BOUND_CAP);auto z=oneesan::gridfp::include_horizontal(m,width,p);if(!z.valid)continue;if(z.blocked)atomic_add_mod(outD+bounded_rank(z.mate,width-1,D_BOUND_DP,D_BOUND_CAP),c);else atomic_add_mod(outM+bounded_rank(z.mate,width,D_BOUND_DP,D_BOUND_CAP),c);}}
__global__ void bounded_block_kernel(const Count*in,Code n,Count*outM,int width,int p){for(Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,st=Code(gridDim.x)*blockDim.x;i<n;i+=st){Count c=in[i];if(!c)continue;MateID b=bounded_unrank(i,width-1,D_BOUND_DP,D_BOUND_CAP);MateID t=oneesan::gridfp::blocked_exclude(b,p);atomic_add_mod(outM+bounded_rank(t,width,D_BOUND_DP,D_BOUND_CAP),c);}}

__global__ void bounded_reverse_main_kernel(const Count*in,const Count*din,Code n,Count*out,int width,int p){
    for(Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,st=Code(gridDim.x)*blockDim.x;i<n;i+=st){
        MateID t=bounded_unrank(i,width,D_BOUND_DP,D_BOUND_CAP); Count acc=in[i]; MateID pred=0; bool have=false;
        switch(oneesan::gridfp::mpair(t,p)){
        case oneesan::gridfp::LR: pred=oneesan::gridfp::msetpair(t,p,oneesan::gridfp::NN);have=true;break;
        case oneesan::gridfp::NR: pred=oneesan::gridfp::msetpair(t,p,oneesan::gridfp::RN);have=true;break;
        case oneesan::gridfp::NL: pred=oneesan::gridfp::msetpair(t,p,oneesan::gridfp::LN);have=true;break;
        default: break;
        }
        if(have){Count c=in[bounded_rank(pred,width,D_BOUND_DP,D_BOUND_CAP)];if(c)acc=(acc>=D_MOD-c)?acc-(D_MOD-c):acc+c;}
        if(oneesan::gridfp::mget(t,p)==oneesan::gridfp::N){MateID b=oneesan::gridfp::mshrink(t,p);Count c=din[bounded_rank(b,width-1,D_BOUND_DP,D_BOUND_CAP)];if(c)acc=(acc>=D_MOD-c)?acc-(D_MOD-c):acc+c;}
        out[i]=acc;
    }
}
__global__ void bounded_forward_block_kernel(const Count*in,Code n,Count*outD,int width,int p){
    for(Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,st=Code(gridDim.x)*blockDim.x;i<n;i+=st){Count c=in[i];if(!c)continue;MateID m=bounded_unrank(i,width,D_BOUND_DP,D_BOUND_CAP);auto z=oneesan::gridfp::include_horizontal(m,width,p);if(z.valid&&z.blocked)atomic_add_mod(outD+bounded_rank(z.mate,width-1,D_BOUND_DP,D_BOUND_CAP),c);}
}

template<int WIDTH>
__device__ __forceinline__ Code rank_drop_n_t(Code src_rank,MateID m,int p){Code a=0,b=0;int h=1;
#pragma unroll
for(int pos=WIDTH-1;pos>p;--pos){MateValue v=mget(m,pos);if(v>N&&allowed(D_MAIN_FIXED,D_MAIN_OCC,pos,N))a+=D_MAIN_DP[pos][h];if(v>R&&h>0&&allowed(D_MAIN_FIXED,D_MAIN_OCC,pos,R))a+=D_MAIN_DP[pos][h-1];int q=pos-1;if(v>N&&allowed(D_BLOCK_FIXED,D_BLOCK_OCC,q,N))b+=D_BLOCK_DP[q][h];if(v>R&&h>0&&allowed(D_BLOCK_FIXED,D_BLOCK_OCC,q,R))b+=D_BLOCK_DP[q][h-1];if(v==R)--h;else if(v==L)++h;}return b>=a?src_rank+(b-a):src_rank-(a-b);}
__global__ void blocked_group_kernel(const Count*in,Code n,Count*out_main,int p){Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;for(;i<n;i+=stride){Count c=in[i];if(!c)continue;MateID sm=factor_unrank_block(i);MateID t=oneesan::gridfp::blocked_exclude(sm,p);Code j=factor_rank_main(t);atomic_add_mod(out_main+j,c);}}
__global__ void main_group_kernel(const Count*in,const MateID*mates,Code n,Count*out_main,Count*out_block,int p){Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;for(;i<n;i+=stride){Count c=in[i];if(!c)continue;MateID m=mates?mates[i]:factor_unrank_main(i);auto z=oneesan::gridfp::include_horizontal(m,TARGET_W,p);if(!z.valid)continue;if(z.blocked)atomic_add_mod(out_block+factor_rank_block(z.mate),c);else atomic_add_mod(out_main+factor_rank_main(z.mate),c);}}

__device__ __forceinline__ Count add_mod_plain(Count a,Count b){Count mod=D_MOD;return a>=mod-b?a-(mod-b):a+b;}

// Reverse (target-gather) form of one Grid-FP update for p>1.  For legal
// targets the main predecessors below are always legal states in the same
// transition-closed occupancy group, so no rank/unrank membership roundtrip
// is required.
__global__ void reverse_main_group_kernel(const Count*in,const Count*din,const MateID*mates,
        Code n,Code dn,Count*out,int p){
    Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;
    for(;i<n;i+=stride){
        MateID t=mates?mates[i]:factor_unrank_main(i); Count acc=in[i];
        MateID pred=0; bool have=false;
        switch(oneesan::gridfp::mpair(t,p)){
        case oneesan::gridfp::LR: pred=oneesan::gridfp::msetpair(t,p,oneesan::gridfp::NN);have=true;break;
        case oneesan::gridfp::NR: pred=oneesan::gridfp::msetpair(t,p,oneesan::gridfp::RN);have=true;break;
        case oneesan::gridfp::NL: pred=oneesan::gridfp::msetpair(t,p,oneesan::gridfp::LN);have=true;break;
        default: break;
        }
        if(have){Count c=in[factor_rank_main(pred)];if(c)acc=add_mod_plain(acc,c);}
        if(oneesan::gridfp::mget(t,p)==oneesan::gridfp::N){
            MateID b=oneesan::gridfp::mshrink(t,p);Count c=din[factor_rank_block(b)];if(c)acc=add_mod_plain(acc,c);
        }
        out[i]=acc;
    }
}


__global__ void forward_block_only_kernel(const Count*in,const MateID*mates,Code n,Count*out_block,int p){
    Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;
    for(;i<n;i+=stride){
        Count c=in[i]; if(!c)continue;
        MateID m=mates?mates[i]:factor_unrank_main(i);
        auto z=oneesan::gridfp::include_horizontal(m,TARGET_W,p);
        if(z.valid&&z.blocked)atomic_add_mod(out_block+factor_rank_block(z.mate),c);
    }
}
__device__ __forceinline__ void reverse_add_main(Count&acc,const Count*in,MateID m){
    Count c=in[factor_rank_main(m)];if(c)acc=add_mod_plain(acc,c);
}
__global__ void reverse_block_group_kernel(const Count*in,Code n,Count*out,Code dn,int p){
    Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;
    for(;i<dn;i+=stride){
        MateID b=factor_unrank_block(i); Count acc=0; auto low=oneesan::gridfp::mget(b,p-1);
        if(low==oneesan::gridfp::R||low==oneesan::gridfp::L){
            // inverse of NR/NL -> shrink(p)
            reverse_add_main(acc,in,oneesan::gridfp::minsert(b,p,oneesan::gridfp::N));
        }else if(low==oneesan::gridfp::N){
            MateID u=oneesan::gridfp::minsert(b,p-1,oneesan::gridfp::N);
            // RL is legal iff the R at the high member of the pair does not
            // cross height zero.
            if(height_before_rank_pos<TARGET_W>(u,p)>0)
                reverse_add_main(acc,in,oneesan::gridfp::msetpair(u,p,oneesan::gridfp::RL));
            // Inverse LL closure.  A candidate occurs whenever the forward
            // closure stack is at depth one and the target carries L there.
            int s=1;
            for(int q=p-2;q>=0&&s>0;--q){
                auto v=oneesan::gridfp::mget(u,q);
                if(v==oneesan::gridfp::L&&s==1){MateID c=oneesan::gridfp::msetpair(u,p,oneesan::gridfp::LL);c=oneesan::gridfp::mset(c,q,oneesan::gridfp::R);reverse_add_main(acc,in,c);}
                if(v==oneesan::gridfp::L)++s;else if(v==oneesan::gridfp::R)--s;
            }
            // Inverse RR closure, symmetrically to the right.
            s=1;
            for(int q=p+1;q<TARGET_W&&s>0;++q){
                auto v=oneesan::gridfp::mget(u,q);
                if(v==oneesan::gridfp::R&&s==1){MateID c=oneesan::gridfp::msetpair(u,p,oneesan::gridfp::RR);c=oneesan::gridfp::mset(c,q,oneesan::gridfp::L);reverse_add_main(acc,in,c);}
                if(v==oneesan::gridfp::L)--s;else if(v==oneesan::gridfp::R)++s;
            }
        }
        out[i]=acc;
    }
}

// Evaluate one reverse Grid-FP step at an arbitrary main target without
// materializing the intermediate vector.  This is used to compose two
// consecutive cell updates.
__device__ __forceinline__ Count reverse_eval_main_step(const Count*in,const Count*din,MateID t,int p){
    Count acc=in[factor_rank_main(t)];
    MateID pred=0; bool have=false;
    switch(oneesan::gridfp::mpair(t,p)){
    case oneesan::gridfp::LR: pred=oneesan::gridfp::msetpair(t,p,oneesan::gridfp::NN);have=true;break;
    case oneesan::gridfp::NR: pred=oneesan::gridfp::msetpair(t,p,oneesan::gridfp::RN);have=true;break;
    case oneesan::gridfp::NL: pred=oneesan::gridfp::msetpair(t,p,oneesan::gridfp::LN);have=true;break;
    default: break;
    }
    if(have){Count c=in[factor_rank_main(pred)];if(c)acc=add_mod_plain(acc,c);}
    if(oneesan::gridfp::mget(t,p)==oneesan::gridfp::N){
        MateID b=oneesan::gridfp::mshrink(t,p);Count c=din[factor_rank_block(b)];if(c)acc=add_mod_plain(acc,c);
    }
    return acc;
}

__device__ __forceinline__ Count reverse_eval_block_step(const Count*in,MateID b,int p){
    Count acc=0; auto low=oneesan::gridfp::mget(b,p-1);
    auto addp=[&](MateID m){Count c=in[factor_rank_main(m)];if(c)acc=add_mod_plain(acc,c);};
    if(low==oneesan::gridfp::R||low==oneesan::gridfp::L){
        addp(oneesan::gridfp::minsert(b,p,oneesan::gridfp::N));
    }else if(low==oneesan::gridfp::N){
        MateID u=oneesan::gridfp::minsert(b,p-1,oneesan::gridfp::N);
        if(height_before_rank_pos<TARGET_W>(u,p)>0)addp(oneesan::gridfp::msetpair(u,p,oneesan::gridfp::RL));
        int s=1;
        for(int q=p-2;q>=0&&s>0;--q){
            auto v=oneesan::gridfp::mget(u,q);
            if(v==oneesan::gridfp::L&&s==1){MateID c=oneesan::gridfp::msetpair(u,p,oneesan::gridfp::LL);c=oneesan::gridfp::mset(c,q,oneesan::gridfp::R);addp(c);}
            if(v==oneesan::gridfp::L)++s;else if(v==oneesan::gridfp::R)--s;
        }
        s=1;
        for(int q=p+1;q<TARGET_W&&s>0;++q){
            auto v=oneesan::gridfp::mget(u,q);
            if(v==oneesan::gridfp::R&&s==1){MateID c=oneesan::gridfp::msetpair(u,p,oneesan::gridfp::RR);c=oneesan::gridfp::mset(c,q,oneesan::gridfp::L);addp(c);}
            if(v==oneesan::gridfp::L)--s;else if(v==oneesan::gridfp::R)++s;
        }
    }
    return acc;
}

// Apply update p and then q=p-1 in one target-gather kernel.  The first
// update is evaluated recursively from the old vectors, so its full M/D
// vectors never touch HBM.
__global__ void reverse2_main_group_kernel(const Count*in,const Count*din,const MateID*mates,
        Code n,Count*out,int p){
    int q=p-1;
    Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;
    for(;i<n;i+=stride){
        MateID t=mates?mates[i]:factor_unrank_main(i);
        Count acc=reverse_eval_main_step(in,din,t,p);
        MateID pred=0; bool have=false;
        switch(oneesan::gridfp::mpair(t,q)){
        case oneesan::gridfp::LR: pred=oneesan::gridfp::msetpair(t,q,oneesan::gridfp::NN);have=true;break;
        case oneesan::gridfp::NR: pred=oneesan::gridfp::msetpair(t,q,oneesan::gridfp::RN);have=true;break;
        case oneesan::gridfp::NL: pred=oneesan::gridfp::msetpair(t,q,oneesan::gridfp::LN);have=true;break;
        default: break;
        }
        if(have){Count c=reverse_eval_main_step(in,din,pred,p);if(c)acc=add_mod_plain(acc,c);}
        if(oneesan::gridfp::mget(t,q)==oneesan::gridfp::N){
            MateID b=oneesan::gridfp::mshrink(t,q);Count c=reverse_eval_block_step(in,b,p);if(c)acc=add_mod_plain(acc,c);
        }
        out[i]=acc;
    }
}

__device__ __forceinline__ void reverse2_add_main(Count&acc,const Count*in,const Count*din,MateID m,int p){
    Count c=reverse_eval_main_step(in,din,m,p);if(c)acc=add_mod_plain(acc,c);
}

__global__ void reverse2_block_group_kernel(const Count*in,const Count*din,const MateID*mates,
        Count*out,Code dn,int p){
    int q=p-1;
    Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;
    for(;i<dn;i+=stride){
        MateID b=mates?mates[i]:factor_unrank_block(i); Count acc=0; auto low=oneesan::gridfp::mget(b,q-1);
        if(low==oneesan::gridfp::R||low==oneesan::gridfp::L){
            reverse2_add_main(acc,in,din,oneesan::gridfp::minsert(b,q,oneesan::gridfp::N),p);
        }else if(low==oneesan::gridfp::N){
            MateID u=oneesan::gridfp::minsert(b,q-1,oneesan::gridfp::N);
            if(height_before_rank_pos<TARGET_W>(u,q)>0)
                reverse2_add_main(acc,in,din,oneesan::gridfp::msetpair(u,q,oneesan::gridfp::RL),p);
            int s=1;
            for(int r=q-2;r>=0&&s>0;--r){
                auto v=oneesan::gridfp::mget(u,r);
                if(v==oneesan::gridfp::L&&s==1){MateID c=oneesan::gridfp::msetpair(u,q,oneesan::gridfp::LL);c=oneesan::gridfp::mset(c,r,oneesan::gridfp::R);reverse2_add_main(acc,in,din,c,p);}
                if(v==oneesan::gridfp::L)++s;else if(v==oneesan::gridfp::R)--s;
            }
            s=1;
            for(int r=q+1;r<TARGET_W&&s>0;++r){
                auto v=oneesan::gridfp::mget(u,r);
                if(v==oneesan::gridfp::R&&s==1){MateID c=oneesan::gridfp::msetpair(u,q,oneesan::gridfp::RR);c=oneesan::gridfp::mset(c,r,oneesan::gridfp::L);reverse2_add_main(acc,in,din,c,p);}
                if(v==oneesan::gridfp::L)--s;else if(v==oneesan::gridfp::R)++s;
            }
        }
        out[i]=acc;
    }
}


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

static PreparedGroup prepare_group(int W,const WindowPlan&wp,int g,Code mc,Code bc,int ng){
    PreparedGroup pg;pg.g=g;
    window_masks(W,wp.p_hi,wp.p_lo,wp.fixed_pos,(uint32_t)g,pg.mf,pg.mo,pg.bf,pg.bo);
    pg.ms=make_spec(W,pg.mf,pg.mo);pg.ds=make_spec(W-1,pg.bf,pg.bo);
    pg.work=2*pg.ms.size+pg.ds.size;
    pg.use_mi=false; pg.use_di=false;
    pg.mi.clear(); pg.di.clear();
    return pg;
}

static void process_group(DeviceCtx&c,int W,const WindowPlan&wp,const PreparedGroup&pg,int threads,size_t target){
    auto t0=std::chrono::steady_clock::now();ck(cudaSetDevice(c.dev),"set worker");
    auto const&ms=pg.ms;auto const&ds=pg.ds;if(!ms.size&&!ds.size)return;
    bool fixLow=wp.p_hi>LOW_LUT_K;uint32_t fmask=fixLow?(pg.mo&((1u<<LOW_LUT_K)-1u)):((pg.mo>>(LOW_LUT_K+1))&((1u<<HIGH_LUT_K)-1u));
    auto fmb=make_factor_main_blocks(fixLow,fmask);auto fdb=make_factor_block_blocks(fixLow,fmask);
    if(fmb.back().end!=ms.size||fdb.back().end!=ds.size){std::cerr<<"factor size mismatch main="<<fmb.back().end<<"/"<<ms.size<<" block="<<fdb.back().end<<"/"<<ds.size<<" fixLow="<<fixLow<<" mask="<<fmask<<"\n";std::exit(20);}
    int fm=(int)fmb.size(),fd=(int)fdb.size(),fl=fixLow?1:0;ck(cudaMemcpyToSymbol(D_F_MAIN_BLOCKS,fmb.data(),fmb.size()*sizeof(FBlock)),"factor main blocks");ck(cudaMemcpyToSymbol(D_F_BLOCK_BLOCKS,fdb.data(),fdb.size()*sizeof(FBlock)),"factor block blocks");ck(cudaMemcpyToSymbol(D_F_MAIN_NBLOCKS,&fm,sizeof(fm)),"factor main n");ck(cudaMemcpyToSymbol(D_F_BLOCK_NBLOCKS,&fd,sizeof(fd)),"factor block n");ck(cudaMemcpyToSymbol(D_F_MASK,&fmask,sizeof(fmask)),"factor mask");ck(cudaMemcpyToSymbol(D_F_FIX_LOW,&fl,sizeof(fl)),"factor mode");
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
    for(int p=wp.p_hi;p>=wp.p_lo;){
        if(p>2 && p-1>=wp.p_lo){
            if(ms.size)reverse2_main_group_kernel<<<bm,threads,0,c.sMain>>>(cur,dcur,useMate?c.dMate:nullptr,ms.size,nxt,p);
            if(ds.size)reverse2_block_group_kernel<<<bd,threads,0,c.sBlock>>>(cur,dcur,nullptr,dnext,ds.size,p);
            p-=2;
        }else{
            if(p>1){
                if(ms.size)reverse_main_group_kernel<<<bm,threads,0,c.sMain>>>(cur,dcur,useMate?c.dMate:nullptr,ms.size,ds.size,nxt,p);
                if(ds.size)reverse_block_group_kernel<<<bd,threads,0,c.sBlock>>>(cur,ms.size,dnext,ds.size,p);
            }else{
                if(ms.size)ck(cudaMemcpyAsync(nxt,cur,size_t(ms.size)*sizeof(Count),cudaMemcpyDeviceToDevice,c.sMain),"p1 identity");
                if(ds.size)ck(cudaMemsetAsync(dnext,0,size_t(ds.size)*sizeof(Count),c.sBlock),"p1 clear next D");
                ck(cudaEventRecord(c.copyDone,c.sMain),"record p1 copy");ck(cudaEventRecord(c.clearDone,c.sBlock),"record p1 clear");
                ck(cudaStreamWaitEvent(c.sMain,c.clearDone,0),"p1 main wait clear");ck(cudaStreamWaitEvent(c.sBlock,c.copyDone,0),"p1 block wait copy");
                if(ms.size)main_group_kernel<<<bm,threads,0,c.sMain>>>(cur,useMate?c.dMate:nullptr,ms.size,nxt,dnext,p);
                if(ds.size)blocked_group_kernel<<<bd,threads,0,c.sBlock>>>(dcur,ds.size,nxt,p);
            }
            --p;
        }
        ck(cudaGetLastError(),"reverse2 bounded transition");
        ck(cudaEventRecord(c.mainDone,c.sMain),"record main");ck(cudaEventRecord(c.blockDone,c.sBlock),"record block");
        ck(cudaStreamWaitEvent(c.sMain,c.blockDone,0),"main wait block");ck(cudaStreamWaitEvent(c.sBlock,c.mainDone,0),"block wait main");
        std::swap(cur,nxt);std::swap(dcur,dnext);
    }
    ck(cudaStreamSynchronize(c.sMain),"main sync");ck(cudaStreamSynchronize(c.sBlock),"block sync");
    if(ms.size){if(pg.use_mi)interval_io_kernel<false,true><<<interval_blocks(pg.mi.size(),threads),threads>>>(cur,c.dIM,pg.mi.size());else scatter_main_kernel<<<bm,threads>>>(cur,ms.size);}
    if(ds.size){if(pg.use_di)interval_io_kernel<true,true><<<interval_blocks(pg.di.size(),threads),threads>>>(dcur,c.dID,pg.di.size());else scatter_block_kernel<<<bd,threads>>>(dcur,ds.size);}
    ck(cudaGetLastError(),"doubleD scatter");ck(cudaDeviceSynchronize(),"group sync");c.groups++;c.active+=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
}


static void run_bounded_prefix_single_gpu(int W,int K,Count* fullMain,int threads){
    if(K<1)return;cudaSetDevice(0);Code dpOld[MAXW+1][MAXW+2]{},dpNew[MAXW+1][MAXW+2]{};
    build_bounded_dp(1,dpOld);Code oldN=dpOld[W][1];Count*cur=nullptr;ck(cudaMalloc(&cur,size_t(oldN)*sizeof(Count)),"bounded cap1");int bo=int(std::min<Code>(65535,(oldN+threads-1)/threads));bounded_fill_one_kernel<<<std::max(1,bo),threads>>>(cur,oldN);ck(cudaDeviceSynchronize(),"bounded fill1");
    for(int cap=2;cap<=K;++cap){build_bounded_dp(cap,dpNew);Code n=dpNew[W][1],dn=dpNew[W-1][1];Count *a=nullptr,*b=nullptr,*d=nullptr,*e=nullptr;ck(cudaMalloc(&a,size_t(n)*sizeof(Count)),"bounded a");ck(cudaMalloc(&b,size_t(n)*sizeof(Count)),"bounded b");ck(cudaMalloc(&d,size_t(dn)*sizeof(Count)),"bounded d");ck(cudaMalloc(&e,size_t(dn)*sizeof(Count)),"bounded e");ck(cudaMemset(a,0,size_t(n)*sizeof(Count)),"bounded zero a");ck(cudaMemset(d,0,size_t(dn)*sizeof(Count)),"bounded zero d");ck(cudaMemcpyToSymbol(D_BOUND_OLD_DP,dpOld,sizeof(dpOld)),"bounded old dp");ck(cudaMemcpyToSymbol(D_BOUND_DP,dpNew,sizeof(dpNew)),"bounded new dp");int oldcap=cap-1;ck(cudaMemcpyToSymbol(D_BOUND_OLD_CAP,&oldcap,sizeof(oldcap)),"bounded old cap");ck(cudaMemcpyToSymbol(D_BOUND_CAP,&cap,sizeof(cap)),"bounded cap");int bm=int(std::min<Code>(65535,(n+threads-1)/threads)),bd=int(std::min<Code>(65535,(dn+threads-1)/threads));int be=int(std::min<Code>(65535,(oldN+threads-1)/threads));bounded_embed_kernel<<<std::max(1,be),threads>>>(cur,oldN,a,W);ck(cudaDeviceSynchronize(),"bounded embed");cudaFree(cur);cur=a;Count*nxt=b,*dc=d,*de=e;
        for(int p=W-1;p>=1;--p){
            if(p>1){ck(cudaMemset(de,0,size_t(dn)*sizeof(Count)),"bounded clear d");bounded_reverse_main_kernel<<<std::max(1,bm),threads>>>(cur,dc,n,nxt,W,p);bounded_forward_block_kernel<<<std::max(1,bm),threads>>>(cur,n,de,W,p);}
            else {ck(cudaMemcpy(nxt,cur,size_t(n)*sizeof(Count),cudaMemcpyDeviceToDevice),"bounded identity");ck(cudaMemset(de,0,size_t(dn)*sizeof(Count)),"bounded clear d");bounded_main_kernel<<<std::max(1,bm),threads>>>(cur,n,nxt,de,W,p);bounded_block_kernel<<<std::max(1,bd),threads>>>(dc,dn,nxt,W,p);}
            ck(cudaDeviceSynchronize(),"bounded step");std::swap(cur,nxt);std::swap(dc,de);}cudaFree(nxt);cudaFree(dc);cudaFree(de);std::memcpy(dpOld,dpNew,sizeof(dpOld));oldN=n;std::cerr<<"bounded prefix cap="<<cap<<" states="<<n<<" block="<<dn<<"\n";
    }
    if(K==1){ck(cudaMemcpyToSymbol(D_BOUND_DP,dpOld,sizeof(dpOld)),"bounded final dp1");int cap=1;ck(cudaMemcpyToSymbol(D_BOUND_CAP,&cap,sizeof(cap)),"bounded final cap1");}
    else {ck(cudaMemcpyToSymbol(D_BOUND_DP,dpOld,sizeof(dpOld)),"bounded final dp");ck(cudaMemcpyToSymbol(D_BOUND_CAP,&K,sizeof(K)),"bounded final cap");}
    int bs=int(std::min<Code>(65535,(oldN+threads-1)/threads));bounded_scatter_full_kernel<<<std::max(1,bs),threads>>>(cur,oldN,W);ck(cudaDeviceSynchronize(),"bounded scatter full");cudaFree(cur);
}


static Count host_pow_mod(Count a,unsigned long long e,Count mod){unsigned long long x=a,r=1;while(e){if(e&1)r=(__uint128_t)r*x%mod;x=(__uint128_t)x*x%mod;e>>=1;}return Count(r);}
static Count host_add_signed(Count acc,Count x,int coeff,Count mod){long long c=coeff;unsigned long long mag=(unsigned long long)(c<0?-c:c);Count z=Count((__uint128_t)x*mag%mod);if(c<0)return acc>=z?acc-z:acc+(mod-z);return acc>=mod-z?acc-(mod-z):acc+z;}
static void row3_apply_row(const Count*in,Count*out,int sym,Count mod){using namespace oneesan::row3auto;std::fill(out,out+DIM,0);const uint16_t*off=sym==0?OFF_N:sym==1?OFF_R:OFF_L;const Tr*tr=sym==0?TR_N:sym==1?TR_R:TR_L;for(int i=0;i<DIM;++i)if(in[i])for(int q=off[i];q<off[i+1];++q)out[tr[q].dst]=host_add_signed(out[tr[q].dst],in[i],tr[q].coeff,mod);}
static void row3_apply_col(int sym,const Count*in,Count*out,Count mod){using namespace oneesan::row3auto;std::fill(out,out+DIM,0);const uint16_t*off=sym==0?OFF_N:sym==1?OFF_R:OFF_L;const Tr*tr=sym==0?TR_N:sym==1?TR_R:TR_L;for(int i=0;i<DIM;++i)for(int q=off[i];q<off[i+1];++q)if(in[tr[q].dst])out[i]=host_add_signed(out[i],in[tr[q].dst],tr[q].coeff,mod);}
static void row4_apply_row(const Count*in,Count*out,int sym,Count mod){using namespace oneesan::row4mod;std::fill(out,out+DIM,0);const uint32_t*off=sym==0?OFF_N:sym==1?OFF_R:OFF_L;const Tr*tr=sym==0?TR_N:sym==1?TR_R:TR_L;for(int i=0;i<DIM;++i)if(in[i])for(uint32_t q=off[i];q<off[i+1];++q)out[tr[q].dst]=Count((out[tr[q].dst]+(__uint128_t)in[i]*tr[q].coeff)%mod);}
static void row4_apply_col(int sym,const Count*in,Count*out,Count mod){using namespace oneesan::row4mod;std::fill(out,out+DIM,0);const uint32_t*off=sym==0?OFF_N:sym==1?OFF_R:OFF_L;const Tr*tr=sym==0?TR_N:sym==1?TR_R:TR_L;for(int i=0;i<DIM;++i)for(uint32_t q=off[i];q<off[i+1];++q)if(in[tr[q].dst])out[i]=Count((out[i]+(__uint128_t)tr[q].coeff*in[tr[q].dst])%mod);}
static std::vector<Count> make_row4_prefix_lut(int len,Count mod){using namespace oneesan::row4mod;std::vector<Count>cur(DIM);cur[0]=1;size_t cnt=1;for(int d=0;d<len;++d){std::vector<Count>nxt(cnt*3*DIM);for(size_t c=0;c<cnt;++c)for(int a=0;a<3;++a)row4_apply_row(&cur[c*DIM],&nxt[(c*3+a)*DIM],a,mod);cur.swap(nxt);cnt*=3;}return cur;}
static std::vector<Count> make_row4_suffix_lut(int len,Count mod){using namespace oneesan::row4mod;std::vector<Count>cur(DIM);for(int i=0;i<DIM;++i)cur[i]=BETA[i];size_t cnt=1,pow3=1;for(int d=0;d<len;++d){std::vector<Count>nxt(cnt*3*DIM);for(int a=0;a<3;++a)for(size_t c=0;c<cnt;++c)row4_apply_col(a,&cur[c*DIM],&nxt[(size_t(a)*pow3+c)*DIM],mod);cur.swap(nxt);cnt*=3;pow3*=3;}return cur;}
static constexpr int R4_BSZ[5]={19,20,12,4,1};
static constexpr int R4_BIDX[5][20]={
 {2,5,7,13,15,20,23,29,35,37,41,44,48,49,51,52,53,54,55,-1},
 {0,1,4,8,10,12,16,18,21,22,26,28,34,38,40,42,43,46,47,50},
 {3,6,9,14,17,24,25,30,32,36,39,45,-1,-1,-1,-1,-1,-1,-1,-1},
 {11,19,27,31,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
 {33,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1}};
static int ternary_prefix_height(size_t code,int len){std::vector<int>d(len);for(int i=len-1;i>=0;--i){d[i]=code%3;code/=3;}int h=1;for(int x:d){if(x==1)--h;else if(x==2)++h;}return h;}
static int ternary_suffix_required_height(size_t code,int len){int delta=0;for(int i=0;i<len;++i){int x=code%3;code/=3;if(x==1)--delta;else if(x==2)++delta;}return -delta;}
static std::vector<Count> compact_row4_lut(const std::vector<Count>&full,int len,bool prefix){using namespace oneesan::row4mod;size_t cnt=full.size()/DIM;std::vector<Count>z(cnt*20);for(size_t c=0;c<cnt;++c){int h=prefix?ternary_prefix_height(c,len):ternary_suffix_required_height(c,len);if(h<0||h>4)continue;for(int q=0;q<R4_BSZ[h];++q)z[c*20+q]=full[c*DIM+R4_BIDX[h][q]];}return z;}

static void row5_apply_row(const Count*in,Count*out,int sym,Count mod){using namespace oneesan::row5mod;std::fill(out,out+DIM,0);const uint32_t*off=sym==0?OFF_N:sym==1?OFF_R:OFF_L;const Tr*tr=sym==0?TR_N:sym==1?TR_R:TR_L;for(int i=0;i<DIM;++i)if(in[i])for(uint32_t q=off[i];q<off[i+1];++q)out[tr[q].dst]=Count((out[tr[q].dst]+(__uint128_t)in[i]*tr[q].coeff)%mod);}
static void row5_apply_col(int sym,const Count*in,Count*out,Count mod){using namespace oneesan::row5mod;std::fill(out,out+DIM,0);const uint32_t*off=sym==0?OFF_N:sym==1?OFF_R:OFF_L;const Tr*tr=sym==0?TR_N:sym==1?TR_R:TR_L;for(int i=0;i<DIM;++i)for(uint32_t q=off[i];q<off[i+1];++q)if(in[tr[q].dst])out[i]=Count((out[i]+(__uint128_t)tr[q].coeff*in[tr[q].dst])%mod);}
static std::vector<Count> make_row5_prefix_lut(int len,Count mod){using namespace oneesan::row5mod;std::vector<Count>cur(DIM);cur[0]=1;size_t cnt=1;for(int d=0;d<len;++d){std::vector<Count>nxt(cnt*3*DIM);for(size_t c=0;c<cnt;++c)for(int a=0;a<3;++a)row5_apply_row(&cur[c*DIM],&nxt[(c*3+a)*DIM],a,mod);cur.swap(nxt);cnt*=3;}return cur;}
static std::vector<Count> make_row5_suffix_lut(int len,Count mod){using namespace oneesan::row5mod;std::vector<Count>cur(DIM);for(int i=0;i<DIM;++i)cur[i]=BETA[i];size_t cnt=1,pow3=1;for(int d=0;d<len;++d){std::vector<Count>nxt(cnt*3*DIM);for(int a=0;a<3;++a)for(size_t c=0;c<cnt;++c)row5_apply_col(a,&cur[c*DIM],&nxt[(size_t(a)*pow3+c)*DIM],mod);cur.swap(nxt);cnt*=3;pow3*=3;}return cur;}
struct Row5RuntimeTr{uint16_t dst;Count coeff;};
static std::array<std::vector<Row5RuntimeTr>,3> G_R5_TR;
static Count G_R5_BETA[176]{};
static Count G_R5_MOD=0;
static uint64_t row5_big_mod(oneesan::row5rat::BigNum z,Count mod){
    uint64_t r64=uint64_t(((__uint128_t(1)<<64)%mod));
    uint64_t v=z.x2%mod;v=(__uint128_t(v)*r64+z.x1)%mod;v=(__uint128_t(v)*r64+z.x0)%mod;return v;
}
static uint64_t row5_pow_mod(uint64_t a,uint64_t e,Count mod){uint64_t r=1;while(e){if(e&1)r=(__uint128_t)r*a%mod;a=(__uint128_t)a*a%mod;e>>=1;}return r;}
static void prepare_row5_runtime(Count mod){
    using namespace oneesan::row5rat;if(G_R5_MOD==mod)return;
    std::vector<Count> dinv(NDEN);for(int i=0;i<NDEN;++i){Count d=Count(row5_big_mod(DEN[i],mod));if(!d)throw std::runtime_error("row5 rational denominator is zero modulo selected prime");dinv[i]=Count(row5_pow_mod(d,uint64_t(mod)-2,mod));}
    const Tr* src[3]={TR_N,TR_R,TR_L};const uint32_t* off[3]={OFF_N,OFF_R,OFF_L};
    for(int a=0;a<3;++a){G_R5_TR[a].resize(off[a][DIM]);for(uint32_t q=0;q<off[a][DIM];++q){auto const&t=src[a][q];uint64_t n=row5_big_mod(t.coeff.num,mod);uint64_t v=(__uint128_t)n*dinv[t.coeff.den]%mod;if(t.coeff.sign<0&&v)v=mod-v;G_R5_TR[a][q]={t.dst,Count(v)};}}
    for(int i=0;i<DIM;++i){int64_t b=BETA[i];uint64_t v=uint64_t(b<0?-b:b)%mod;G_R5_BETA[i]=Count(b<0&&v?mod-v:v);}G_R5_MOD=mod;
}
static constexpr int R5_BSZ[6]={51,61,40,18,5,1};
static constexpr int R5_BIDX[6][61]={
 {2,5,7,13,15,20,23,29,35,37,42,45,51,56,59,62,64,73,79,81,96,98,103,106,112,116,119,122,124,132,136,138,144,147,149,150,154,155,158,161,164,166,167,168,169,170,171,172,173,174,175,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
 {0,1,4,8,10,12,16,18,21,22,26,28,34,38,40,43,44,48,50,57,58,61,65,67,70,72,78,82,84,90,95,99,101,104,105,109,111,117,118,121,125,127,130,131,135,139,141,143,145,146,148,151,152,153,156,157,159,160,162,163,165},
 {3,6,9,14,17,24,25,30,32,36,39,46,47,52,54,60,63,66,69,74,76,80,83,87,89,97,100,107,108,113,115,120,123,126,129,133,134,137,140,142,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
 {11,19,27,31,41,49,53,68,71,75,85,86,91,93,102,110,114,128,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
 {33,55,77,88,92,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
 {94,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1}};
static std::vector<Count> compact_row5_lut(const std::vector<Count>&full,int len,bool prefix){using namespace oneesan::row5mod;size_t cnt=full.size()/DIM;std::vector<Count>z(cnt*61);for(size_t c=0;c<cnt;++c){int h=prefix?ternary_prefix_height(c,len):ternary_suffix_required_height(c,len);if(h<0||h>5)continue;for(int q=0;q<R5_BSZ[h];++q)z[c*61+q]=full[c*DIM+R5_BIDX[h][q]];}return z;}

static inline Count row5_mul_mod_host(Count a,Count b){
    if(G_R5_MOD==4294967291u){uint64_t z=uint64_t(a)*b;uint64_t t=uint32_t(z)+(z>>32)*5ULL;t=uint32_t(t)+(t>>32)*5ULL;if(t>=4294967291ULL)t-=4294967291ULL;if(t>=4294967291ULL)t-=4294967291ULL;return Count(t);}
    return Count(uint64_t(a)*b%G_R5_MOD);
}
static inline Count row5_add_mod_host(Count a,Count b){return a>=G_R5_MOD-b?a-(G_R5_MOD-b):a+b;}
static const std::array<int,176>& row5_local_index(){
    static const std::array<int,176> idx=[](){std::array<int,176>x{};x.fill(-1);for(int h=0;h<6;++h)for(int q=0;q<R5_BSZ[h];++q)x[R5_BIDX[h][q]]=q;return x;}();
    return idx;
}
static inline int row5_delta(int sym){return sym==0?0:sym==1?-1:1;}
static void row5_apply_row_compact(const Count*in,Count*out,int h,int sym){
    int h2=h+row5_delta(sym);if(h<0||h>5||h2<0||h2>5)return;using namespace oneesan::row5rat;
    const uint32_t*off=sym==0?OFF_N:sym==1?OFF_R:OFF_L;auto const&tr=G_R5_TR[sym];const auto&li=row5_local_index();
    for(int q=0;q<R5_BSZ[h];++q){Count v=in[q];if(!v)continue;int i=R5_BIDX[h][q];for(uint32_t k=off[i];k<off[i+1];++k){int d=li[tr[k].dst];Count z=row5_mul_mod_host(v,tr[k].coeff);out[d]=row5_add_mod_host(out[d],z);}}
}
static void row5_apply_col_compact(const Count*in,Count*out,int hdst,int sym){
    int hsrc=hdst-row5_delta(sym);if(hdst<0||hdst>5||hsrc<0||hsrc>5)return;using namespace oneesan::row5rat;
    const uint32_t*off=sym==0?OFF_N:sym==1?OFF_R:OFF_L;auto const&tr=G_R5_TR[sym];const auto&li=row5_local_index();
    for(int q=0;q<R5_BSZ[hsrc];++q){int i=R5_BIDX[hsrc][q];Count acc=0;for(uint32_t k=off[i];k<off[i+1];++k){Count v=in[li[tr[k].dst]];if(v)acc=row5_add_mod_host(acc,row5_mul_mod_host(tr[k].coeff,v));}out[q]=acc;}
}
template<class F> static void row5_parallel_for(size_t n,F&&fn){
    unsigned hc=std::thread::hardware_concurrency();int nt=hc?int(hc/2):4;if(const char*e=std::getenv("GRIDFP_ROW5_HOST_THREADS"))nt=std::max(1,std::atoi(e));
    if(n<2048||nt<=1){for(size_t i=0;i<n;++i)fn(i);return;}nt=std::min<int>(nt,int(n));std::vector<std::thread>ts;ts.reserve(nt);
    for(int t=0;t<nt;++t){size_t lo=n*size_t(t)/nt,hi=n*size_t(t+1)/nt;ts.emplace_back([&,lo,hi]{for(size_t i=lo;i<hi;++i)fn(i);});}for(auto&t:ts)t.join();
}
static std::vector<Count> make_row5_prefix_lut_compact_direct(int len){
    constexpr int S=61;std::vector<Count>cur(S);const auto&li=row5_local_index();cur[li[0]]=1;size_t cnt=1;
    std::vector<int8_t> height(1,1);
    for(int d=0;d<len;++d){std::vector<Count>nxt(cnt*3*S);std::vector<int8_t> nh(cnt*3,-127);row5_parallel_for(cnt,[&](size_t c){int h=height[c];if(h<0||h>5)return;for(int a=0;a<3;++a){int h2=h+row5_delta(a);nh[c*3+a]=(h2>=0&&h2<=5)?int8_t(h2):int8_t(-127);row5_apply_row_compact(&cur[c*S],&nxt[(c*3+a)*S],h,a);}});cur.swap(nxt);height.swap(nh);cnt*=3;}return cur;
}
static std::vector<Count> make_row5_suffix_lut_compact_direct(int len){
    constexpr int S=61;std::vector<Count>cur(S);const auto&li=row5_local_index();for(int i=0;i<176;++i)if(G_R5_BETA[i])cur[li[i]]=G_R5_BETA[i];size_t cnt=1;
    std::vector<int8_t> height(1,0);
    for(int d=0;d<len;++d){std::vector<Count>nxt(cnt*3*S);std::vector<int8_t> nh(cnt*3,-127);row5_parallel_for(cnt,[&](size_t c){int h=height[c];if(h<0||h>5)return;for(int a=0;a<3;++a){int h2=h-row5_delta(a);nh[size_t(a)*cnt+c]=(h2>=0&&h2<=5)?int8_t(h2):int8_t(-127);row5_apply_col_compact(&cur[c*S],&nxt[(size_t(a)*cnt+c)*S],h,a);}});cur.swap(nxt);height.swap(nh);cnt*=3;}return cur;
}

static Code row5_bounded_to_full_rank_host(Code brank,int W,const Code dp[MAXW+1][MAXW+2]){
    Code r=brank,g=0;int h=1;for(int pos=W-1;pos>=0;--pos){Code a=dp[pos][h];if(r<a)continue;r-=a;if(h>0){a=dp[pos][h-1];if(r<a){g+=H_DP[pos][h];--h;continue;}r-=a;g+=H_DP[pos][h]+H_DP[pos][h-1];++h;}else{g+=H_DP[pos][h];++h;}}return g;
}
static Code row5_lower_bound_full_rank(Code n,Code target,int W,const Code dp[MAXW+1][MAXW+2]){Code lo=0,hi=n;while(lo<hi){Code m=lo+(hi-lo)/2;if(row5_bounded_to_full_rank_host(m,W,dp)<target)lo=m+1;else hi=m;}return lo;}


struct PackedRow5Lut { std::vector<Count> data; std::vector<uint32_t> idx; };
static int row5_prefix_code_height(size_t code,int len){
    int dig[32];for(int i=len-1;i>=0;--i){dig[i]=int(code%3);code/=3;}int h=1;
    for(int i=0;i<len;++i){h+=row5_delta(dig[i]);if(h<0||h>5)return -1;}return h;
}
static int row5_suffix_code_height(size_t code,int len){
    int h=0;for(int i=0;i<len;++i){int a=int(code%3);code/=3;h-=row5_delta(a);if(h<0||h>5)return -1;}return h;
}
static PackedRow5Lut pack_row5_dense_lut(const std::vector<Count>&dense,int len,bool prefix){
    constexpr int S=61;size_t cnt=dense.size()/S;PackedRow5Lut z;z.idx.assign(cnt,0xffffffffu);
    std::vector<uint8_t> hs(cnt,255);std::vector<uint32_t> off(cnt+1);uint64_t total=0;
    for(size_t c=0;c<cnt;++c){int h=prefix?row5_prefix_code_height(c,len):row5_suffix_code_height(c,len);if(h>=0){hs[c]=uint8_t(h);z.idx[c]=uint32_t(total);total+=R5_BSZ[h];if(total>0xffffffffULL)throw std::runtime_error("row5 packed LUT exceeds uint32 offset");}off[c+1]=uint32_t(total);}
    z.data.resize(size_t(total));
    row5_parallel_for(cnt,[&](size_t c){if(hs[c]==255)return;int h=hs[c],n=R5_BSZ[h];std::copy_n(dense.data()+c*S,n,z.data.data()+z.idx[c]);});
    return z;
}

static void init_direct_row5_lut_full_multi_gpu(int W,Count mod,int threads,Count**fullMain,const std::vector<Code>&mainLen,Code mainChunk,int ng){
    prepare_row5_runtime(mod);Code dp[MAXW+1][MAXW+2]{};build_bounded_dp(5,dp);Code n=dp[W][1];constexpr int lo=TARGET_W/2,hi=TARGET_W-lo;
    auto t0=std::chrono::steady_clock::now();std::vector<Count>prefDense,suffDense;std::thread tp([&]{prefDense=make_row5_prefix_lut_compact_direct(hi);});std::thread ts([&]{suffDense=make_row5_suffix_lut_compact_direct(lo);});tp.join();ts.join();auto pref=pack_row5_dense_lut(prefDense,hi,true);auto suff=pack_row5_dense_lut(suffDense,lo,false);prefDense.clear();prefDense.shrink_to_fit();suffDense.clear();suffDense.shrink_to_fit();double host_s=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
    std::vector<Code> cut(ng+1);cut[0]=0;cut[ng]=n;for(int d=1;d<ng;++d)cut[d]=row5_lower_bound_full_rank(n,Code(d)*mainChunk,W,dp);
    Code mn=~Code(0),mx=0;for(int d=0;d<ng;++d){Code z=cut[d+1]-cut[d];mn=std::min(mn,z);mx=std::max(mx,z);Code fb=Code(d)*mainChunk,first=z?row5_bounded_to_full_rank_host(cut[d],W,dp):fb,last=z?row5_bounded_to_full_rank_host(cut[d+1]-1,W,dp):fb;std::cerr<<"row5 shard gpu="<<d<<" bounded=["<<cut[d]<<","<<cut[d+1]<<") states="<<z<<" full=["<<first<<","<<last<<"] local_cap="<<mainLen[d]<<"\n";}
    int lanes=4;if(const char*e=std::getenv("GRIDFP_ROW5_LANES"))lanes=std::atoi(e);else if(const char*e=std::getenv("GRIDFP_ROW5_WARP"))lanes=std::atoi(e)?32:0;
    std::vector<double> gpu_s(ng);std::vector<std::thread> workers;workers.reserve(ng);auto g0=std::chrono::steady_clock::now();
    for(int d=0;d<ng;++d)workers.emplace_back([&,d]{ck(cudaSetDevice(d),"row5 multi set");ck(cudaMemcpyToSymbol(D_BOUND_DP,dp,sizeof(dp)),"row5 lut dp");int cap=5;ck(cudaMemcpyToSymbol(D_BOUND_CAP,&cap,sizeof(cap)),"row5 lut cap");Count *dpref=nullptr,*dsuff=nullptr;uint32_t *dpidx=nullptr,*dsidx=nullptr;
        ck(cudaMalloc(&dpref,pref.data.size()*sizeof(Count)),"row5 pref");ck(cudaMalloc(&dsuff,suff.data.size()*sizeof(Count)),"row5 suff");
        ck(cudaMalloc(&dpidx,pref.idx.size()*sizeof(uint32_t)),"row5 pref idx");ck(cudaMalloc(&dsidx,suff.idx.size()*sizeof(uint32_t)),"row5 suff idx");
        ck(cudaMemcpy(dpref,pref.data.data(),pref.data.size()*sizeof(Count),cudaMemcpyHostToDevice),"row5 pref copy");ck(cudaMemcpy(dsuff,suff.data.data(),suff.data.size()*sizeof(Count),cudaMemcpyHostToDevice),"row5 suff copy");
        ck(cudaMemcpy(dpidx,pref.idx.data(),pref.idx.size()*sizeof(uint32_t),cudaMemcpyHostToDevice),"row5 pref idx copy");ck(cudaMemcpy(dsidx,suff.idx.data(),suff.idx.size()*sizeof(uint32_t),cudaMemcpyHostToDevice),"row5 suff idx copy");
        ck(cudaMemcpyToSymbol(D_R5_PREF,&dpref,sizeof(dpref)),"row5 pref ptr");ck(cudaMemcpyToSymbol(D_R5_SUFF,&dsuff,sizeof(dsuff)),"row5 suff ptr");ck(cudaMemcpyToSymbol(D_R5_PREF_IDX,&dpidx,sizeof(dpidx)),"row5 pref idx ptr");ck(cudaMemcpyToSymbol(D_R5_SUFF_IDX,&dsidx,sizeof(dsidx)),"row5 suff idx ptr");Code cnt=cut[d+1]-cut[d],base=Code(d)*mainChunk;int b=lanes?65535:int(std::min<Code>(65535,(cnt+threads-1)/threads));auto q0=std::chrono::steady_clock::now();if(cnt){if(lanes==32&&ng==1)bounded_fill_row5_full_warp_kernel<<<std::max(1,b),threads>>>(fullMain[d],n,W);else if(lanes==16)bounded_fill_row5_full_group_range_kernel<16><<<std::max(1,b),threads>>>(fullMain[d],cut[d],cut[d+1],base);else if(lanes==8)bounded_fill_row5_full_group_range_kernel<8><<<std::max(1,b),threads>>>(fullMain[d],cut[d],cut[d+1],base);else if(lanes==4)bounded_fill_row5_full_group_range_kernel<4><<<std::max(1,b),threads>>>(fullMain[d],cut[d],cut[d+1],base);else bounded_fill_row5_full_range_kernel<<<std::max(1,b),threads>>>(fullMain[d],cut[d],cut[d+1],base);}ck(cudaDeviceSynchronize(),"row5 lut init");gpu_s[d]=std::chrono::duration<double>(std::chrono::steady_clock::now()-q0).count();cudaFree(dpref);cudaFree(dsuff);cudaFree(dpidx);cudaFree(dsidx);});
    for(auto&t:workers)t.join();double gpu_wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-g0).count(),gmax=*std::max_element(gpu_s.begin(),gpu_s.end());
    std::cerr<<"direct row5 full states="<<n<<" lut_host_s="<<host_s<<" kernel_max_s="<<gmax<<" gpu_phase_wall_s="<<gpu_wall<<" pref_mib="<<(pref.data.size()*sizeof(Count)>>20)<<" suff_mib="<<(suff.data.size()*sizeof(Count)>>20)<<" pref_idx_mib="<<(pref.idx.size()*sizeof(uint32_t)>>20)<<" suff_idx_mib="<<(suff.idx.size()*sizeof(uint32_t)>>20)<<" shard_min="<<mn<<" shard_max="<<mx<<"\n";
}


struct Row6Layout { std::array<std::array<int,182>,7> bidx{}; std::array<int,558> li{}; Row6Layout(){for(auto&a:bidx)a.fill(-1);li.fill(-1);int n[7]{};for(int i=0;i<558;++i){int h=oneesan::row6mod::HEIGHT[i];li[i]=n[h];bidx[h][n[h]++]=i;}const int w[7]={141,182,135,68,25,6,1};for(int h=0;h<7;++h)if(n[h]!=w[h])throw std::runtime_error("row6 layout mismatch");}};
static const Row6Layout& row6_layout(){static Row6Layout z;return z;}
static inline int row6_delta(int sym){return sym==0?0:sym==1?-1:1;}
static inline Count row6_add_host(Count a,Count b){constexpr Count m=oneesan::row6mod::MOD;return a>=m-b?a-(m-b):a+b;}
static inline Count row6_mul_host(Count a,Count b){return Count(uint64_t(a)*b%oneesan::row6mod::MOD);}
static void row6_apply_row_compact(const Count*in,Count*out,int h,int sym){
    using namespace oneesan::row6mod;int h2=h+row6_delta(sym);if(h<0||h>6||h2<0||h2>6)return;const uint32_t*off=sym==0?OFF_N:sym==1?OFF_R:OFF_L;const Tr*tr=sym==0?TR_N:sym==1?TR_R:TR_L;auto const&lay=row6_layout();
    for(int q=0;q<r6_block_size(h);++q){Count v=in[q];if(!v)continue;int i=lay.bidx[h][q];for(uint32_t k=off[i];k<off[i+1];++k){int d=lay.li[tr[k].dst];out[d]=row6_add_host(out[d],row6_mul_host(v,tr[k].coeff));}}
}
static void row6_apply_col_compact(const Count*in,Count*out,int hdst,int sym){
    using namespace oneesan::row6mod;int hsrc=hdst-row6_delta(sym);if(hdst<0||hdst>6||hsrc<0||hsrc>6)return;const uint32_t*off=sym==0?OFF_N:sym==1?OFF_R:OFF_L;const Tr*tr=sym==0?TR_N:sym==1?TR_R:TR_L;auto const&lay=row6_layout();
    for(int q=0;q<r6_block_size(hsrc);++q){int i=lay.bidx[hsrc][q];Count acc=0;for(uint32_t k=off[i];k<off[i+1];++k){Count v=in[lay.li[tr[k].dst]];if(v)acc=row6_add_host(acc,row6_mul_host(tr[k].coeff,v));}out[q]=acc;}
}
template<class F> static void row6_parallel_for(size_t n,F&&fn){unsigned hc=std::thread::hardware_concurrency();int nt=hc?int(hc/2):4;if(const char*e=std::getenv("GRIDFP_ROW6_HOST_THREADS"))nt=std::max(1,std::atoi(e));if(n<2048||nt<=1){for(size_t i=0;i<n;++i)fn(i);return;}nt=std::min<int>(nt,int(n));std::vector<std::thread>ts;for(int t=0;t<nt;++t){size_t lo=n*size_t(t)/nt,hi=n*size_t(t+1)/nt;ts.emplace_back([&,lo,hi]{for(size_t i=lo;i<hi;++i)fn(i);});}for(auto&t:ts)t.join();}
static std::vector<Count> make_row6_prefix_lut_dense(int len){constexpr int S=182;auto const&lay=row6_layout();std::vector<Count>cur(S);cur[lay.li[0]]=1;size_t cnt=1;std::vector<int8_t> height(1,1);for(int d=0;d<len;++d){std::vector<Count>nxt(cnt*3*S);std::vector<int8_t>nh(cnt*3,-127);row6_parallel_for(cnt,[&](size_t c){int h=height[c];if(h<0||h>6)return;for(int a=0;a<3;++a){int h2=h+row6_delta(a);nh[c*3+a]=(h2>=0&&h2<=6)?int8_t(h2):int8_t(-127);row6_apply_row_compact(&cur[c*S],&nxt[(c*3+a)*S],h,a);}});cur.swap(nxt);height.swap(nh);cnt*=3;}return cur;}
static std::vector<Count> make_row6_suffix_lut_dense(int len){using namespace oneesan::row6mod;constexpr int S=182;auto const&lay=row6_layout();std::vector<Count>cur(S);for(int i=0;i<DIM;++i)if(BETA[i]){if(HEIGHT[i]!=0)throw std::runtime_error("row6 beta height");cur[lay.li[i]]=BETA[i];}size_t cnt=1;std::vector<int8_t>height(1,0);for(int d=0;d<len;++d){std::vector<Count>nxt(cnt*3*S);std::vector<int8_t>nh(cnt*3,-127);row6_parallel_for(cnt,[&](size_t c){int h=height[c];if(h<0||h>6)return;for(int a=0;a<3;++a){int h2=h-row6_delta(a);nh[size_t(a)*cnt+c]=(h2>=0&&h2<=6)?int8_t(h2):int8_t(-127);row6_apply_col_compact(&cur[c*S],&nxt[(size_t(a)*cnt+c)*S],h,a);}});cur.swap(nxt);height.swap(nh);cnt*=3;}return cur;}
struct PackedRow6Lut { std::vector<Count> data; std::vector<uint32_t> idx; };
static int row6_prefix_code_height(size_t code,int len){int d[32];for(int i=len-1;i>=0;--i){d[i]=code%3;code/=3;}int h=1;for(int i=0;i<len;++i){h+=row6_delta(d[i]);if(h<0||h>6)return -1;}return h;}
static int row6_suffix_code_height(size_t code,int len){int h=0;for(int i=0;i<len;++i){int a=code%3;code/=3;h-=row6_delta(a);if(h<0||h>6)return -1;}return h;}
static PackedRow6Lut pack_row6_dense(const std::vector<Count>&dense,int len,bool prefix){constexpr int S=182;size_t cnt=dense.size()/S;PackedRow6Lut z;z.idx.assign(cnt,0xffffffffu);std::vector<uint8_t>hs(cnt,255);uint64_t total=0;for(size_t c=0;c<cnt;++c){int h=prefix?row6_prefix_code_height(c,len):row6_suffix_code_height(c,len);if(h>=0){hs[c]=h;z.idx[c]=uint32_t(total);total+=r6_block_size(h);if(total>0xffffffffULL)throw std::runtime_error("row6 packed offset overflow");}}z.data.resize(total);row6_parallel_for(cnt,[&](size_t c){if(hs[c]==255)return;std::copy_n(dense.data()+c*S,r6_block_size(hs[c]),z.data.data()+z.idx[c]);});return z;}
static void init_direct_row6_lut_full_multi_gpu(int W,Count mod,int threads,Count**fullMain,const std::vector<Code>&mainLen,Code mainChunk,int ng){
    using namespace oneesan::row6mod;if(mod!=MOD)throw std::runtime_error("row6 GPU-level-packed prototype requires modulus 1000000007");Code dp[MAXW+1][MAXW+2]{};build_bounded_dp(6,dp);Code n=dp[W][1];constexpr int lo=TARGET_W/2,hi=TARGET_W-lo;auto t0=std::chrono::steady_clock::now();auto const&lay=row6_layout();
    std::vector<Count>mat,beta(r6_block_size(0));uint32_t moff[3][7]{};const uint32_t*offs[3]={OFF_N,OFF_R,OFF_L};const Tr*trs[3]={TR_N,TR_R,TR_L};for(int sym=0;sym<3;++sym)for(int h=0;h<=6;++h){int h2=h+row6_delta(sym);if(h2<0||h2>6)continue;int ns=r6_block_size(h),nd=r6_block_size(h2);moff[sym][h]=uint32_t(mat.size());mat.resize(mat.size()+size_t(ns)*nd,0);Count*M=mat.data()+moff[sym][h];for(int a=0;a<ns;++a){int gi=lay.bidx[h][a];for(uint32_t q=offs[sym][gi];q<offs[sym][gi+1];++q){int gd=trs[sym][q].dst;if(HEIGHT[gd]!=h2)throw std::runtime_error("row6 transition violates height grading");int d=lay.li[gd];M[size_t(a)*nd+d]=row6_add_host(M[size_t(a)*nd+d],trs[sym][q].coeff);}}}for(int i=0;i<DIM;++i)if(BETA[i]){if(HEIGHT[i]!=0)throw std::runtime_error("row6 beta height");beta[lay.li[i]]=row6_add_host(beta[lay.li[i]],BETA[i]);}int initLocal=lay.li[0];
    auto make_idx=[&](int len,bool prefix){size_t codes=1;for(int i=0;i<len;++i)codes*=3;std::vector<uint32_t>idx(codes,0xffffffffu);uint64_t total=0;for(size_t c=0;c<codes;++c){int h=prefix?row6_prefix_code_height(c,len):row6_suffix_code_height(c,len);if(h>=0){if(total>0xffffffffULL)throw std::runtime_error("row6 packed offset overflow");idx[c]=uint32_t(total);total+=r6_block_size(h);}}if(total>0xffffffffULL)throw std::runtime_error("row6 packed table too large");return std::pair<std::vector<uint32_t>,uint64_t>(std::move(idx),total);};std::pair<std::vector<uint32_t>,uint64_t>pi,si;std::thread ixp([&]{pi=make_idx(hi,true);}),ixs([&]{si=make_idx(lo,false);});ixp.join();ixs.join();double host_s=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
    std::vector<Code>cut(ng+1);cut[0]=0;cut[ng]=n;for(int d=1;d<ng;++d)cut[d]=row5_lower_bound_full_rank(n,Code(d)*mainChunk,W,dp);int lanes=8;if(const char*e=std::getenv("GRIDFP_ROW6_LANES"))lanes=std::atoi(e);int rt=128;if(const char*e=std::getenv("GRIDFP_ROW6_LEVEL_THREADS"))rt=std::max(32,std::atoi(e));std::vector<double>table_s(ng),init_s(ng);std::vector<std::thread>workers;auto g0=std::chrono::steady_clock::now();
    for(int dev=0;dev<ng;++dev)workers.emplace_back([&,dev]{ck(cudaSetDevice(dev),"r6 levelpack set");ck(cudaMemcpyToSymbol(D_BOUND_DP,dp,sizeof(dp)),"r6 levelpack dp");int cap=6;ck(cudaMemcpyToSymbol(D_BOUND_CAP,&cap,sizeof(cap)),"r6 levelpack cap");ck(cudaMemcpyToSymbol(D_R6_GPU_MAT_OFF,moff,sizeof(moff)),"r6 levelpack mat off");Count*dmat=nullptr,*dbeta=nullptr;uint32_t*dpidx=nullptr,*dsidx=nullptr;ck(cudaMalloc(&dmat,mat.size()*4),"r6 levelpack mat");ck(cudaMalloc(&dbeta,beta.size()*4),"r6 levelpack beta");ck(cudaMalloc(&dpidx,pi.first.size()*4),"r6 levelpack pidx");ck(cudaMalloc(&dsidx,si.first.size()*4),"r6 levelpack sidx");ck(cudaMemcpy(dmat,mat.data(),mat.size()*4,cudaMemcpyHostToDevice),"r6 levelpack mat cp");ck(cudaMemcpy(dbeta,beta.data(),beta.size()*4,cudaMemcpyHostToDevice),"r6 levelpack beta cp");ck(cudaMemcpy(dpidx,pi.first.data(),pi.first.size()*4,cudaMemcpyHostToDevice),"r6 levelpack pidx cp");ck(cudaMemcpy(dsidx,si.first.data(),si.first.size()*4,cudaMemcpyHostToDevice),"r6 levelpack sidx cp");auto tb=std::chrono::steady_clock::now();
        auto build_pref=[&](){Code parents=1;Count*cur=nullptr;int8_t*hp=nullptr;ck(cudaMalloc(&cur,size_t(182)*4),"r6 pref base");ck(cudaMemset(cur,0,size_t(182)*4),"r6 pref base zero");Count one=1;ck(cudaMemcpy(cur+initLocal,&one,4,cudaMemcpyHostToDevice),"r6 pref one");ck(cudaMalloc(&hp,1),"r6 pref h");int8_t h1=1;ck(cudaMemcpy(hp,&h1,1,cudaMemcpyHostToDevice),"r6 pref h1");for(int lev=0;lev<hi;++lev){Code children=parents*3;Count*nxt=nullptr;int8_t*hn=nullptr;ck(cudaMalloc(&nxt,size_t(182)*children*4),"r6 pref next");ck(cudaMalloc(&hn,size_t(children)),"r6 pref hn");int bl=int(std::min<Code>(65535,(children+rt-1)/rt));r6_prefix_level_kernel<<<std::max(1,bl),rt>>>(parents,cur,hp,nxt,hn,dmat);ck(cudaGetLastError(),"r6 pref level launch");cudaFree(cur);cudaFree(hp);cur=nxt;hp=hn;parents=children;}Count*packed=nullptr;ck(cudaMalloc(&packed,size_t(pi.second)*4),"r6 pref packed");int bl=int(std::min<Code>(65535,(parents+255)/256));r6_pack_level_kernel<<<std::max(1,bl),256>>>(parents,cur,hp,dpidx,packed);ck(cudaGetLastError(),"r6 pref pack launch");cudaFree(cur);cudaFree(hp);return packed;};
        auto build_suff=[&](){Code parents=1;Count*cur=nullptr;int8_t*hp=nullptr;ck(cudaMalloc(&cur,size_t(182)*4),"r6 suff base");ck(cudaMemset(cur,0,size_t(182)*4),"r6 suff base zero");ck(cudaMemcpy(cur,dbeta,size_t(r6_block_size(0))*4,cudaMemcpyDeviceToDevice),"r6 suff beta");ck(cudaMalloc(&hp,1),"r6 suff h");int8_t h0=0;ck(cudaMemcpy(hp,&h0,1,cudaMemcpyHostToDevice),"r6 suff h0");for(int lev=0;lev<lo;++lev){Code children=parents*3;Count*nxt=nullptr;int8_t*hn=nullptr;ck(cudaMalloc(&nxt,size_t(182)*children*4),"r6 suff next");ck(cudaMalloc(&hn,size_t(children)),"r6 suff hn");int bl=int(std::min<Code>(65535,(children+rt-1)/rt));r6_suffix_level_kernel<<<std::max(1,bl),rt>>>(parents,cur,hp,nxt,hn,dmat);ck(cudaGetLastError(),"r6 suff level launch");cudaFree(cur);cudaFree(hp);cur=nxt;hp=hn;parents=children;}Count*packed=nullptr;ck(cudaMalloc(&packed,size_t(si.second)*4),"r6 suff packed");int bl=int(std::min<Code>(65535,(parents+255)/256));r6_pack_level_kernel<<<std::max(1,bl),256>>>(parents,cur,hp,dsidx,packed);ck(cudaGetLastError(),"r6 suff pack launch");cudaFree(cur);cudaFree(hp);return packed;};
        Count*P=build_pref();Count*S=build_suff();ck(cudaDeviceSynchronize(),"r6 levelpack tables");table_s[dev]=std::chrono::duration<double>(std::chrono::steady_clock::now()-tb).count();ck(cudaMemcpyToSymbol(D_R6_PREF,&P,sizeof(P)),"r6 levelpack pref ptr");ck(cudaMemcpyToSymbol(D_R6_SUFF,&S,sizeof(S)),"r6 levelpack suff ptr");ck(cudaMemcpyToSymbol(D_R6_PREF_IDX,&dpidx,sizeof(dpidx)),"r6 levelpack pidx ptr");ck(cudaMemcpyToSymbol(D_R6_SUFF_IDX,&dsidx,sizeof(dsidx)),"r6 levelpack sidx ptr");Code cnt=cut[dev+1]-cut[dev],base=Code(dev)*mainChunk;int blocks=lanes?65535:int(std::min<Code>(65535,(cnt+threads-1)/threads));auto ib=std::chrono::steady_clock::now();if(cnt){if(lanes==16)bounded_fill_row6_full_group_range_kernel<16><<<std::max(1,blocks),threads>>>(fullMain[dev],cut[dev],cut[dev+1],base);else if(lanes==8)bounded_fill_row6_full_group_range_kernel<8><<<std::max(1,blocks),threads>>>(fullMain[dev],cut[dev],cut[dev+1],base);else if(lanes==4)bounded_fill_row6_full_group_range_kernel<4><<<std::max(1,blocks),threads>>>(fullMain[dev],cut[dev],cut[dev+1],base);else bounded_fill_row6_full_range_kernel<<<std::max(1,blocks),threads>>>(fullMain[dev],cut[dev],cut[dev+1],base);}ck(cudaDeviceSynchronize(),"r6 levelpack init");init_s[dev]=std::chrono::duration<double>(std::chrono::steady_clock::now()-ib).count();cudaFree(P);cudaFree(S);cudaFree(dpidx);cudaFree(dsidx);cudaFree(dmat);cudaFree(dbeta);});for(auto&t:workers)t.join();double gw=std::chrono::duration<double>(std::chrono::steady_clock::now()-g0).count(),tm=*std::max_element(table_s.begin(),table_s.end()),im=*std::max_element(init_s.begin(),init_s.end());std::cerr<<"direct row6 GPU-level-packed states="<<n<<" lut_host_s="<<host_s<<" table_kernel_max_s="<<tm<<" init_kernel_max_s="<<im<<" gpu_phase_wall_s="<<gw<<" pref_mib="<<(pi.second*4>>20)<<" suff_mib="<<(si.second*4>>20)<<" pidx_mib="<<(pi.first.size()*4>>20)<<" sidx_mib="<<(si.first.size()*4>>20)<<" mat_kib="<<(mat.size()*4>>10)<<" level_threads="<<rt<<"\n";
}

static Count* build_direct_row4_lut_compact_single_gpu(int W,Count mod,int threads,Code&outN,Code outDp[MAXW+1][MAXW+2]){
    using namespace oneesan::row4mod;if(mod!=MOD){std::cerr<<"row4 LUT prototype only supports modulus "<<MOD<<"\n";return nullptr;}cudaSetDevice(0);Code dp[MAXW+1][MAXW+2]{};build_bounded_dp(4,dp);outN=dp[W][1];std::memcpy(outDp,dp,sizeof(dp));ck(cudaMemcpyToSymbol(D_BOUND_DP,dp,sizeof(dp)),"row4 lut dp");int cap=4;ck(cudaMemcpyToSymbol(D_BOUND_CAP,&cap,sizeof(cap)),"row4 lut cap");constexpr int lo=TARGET_W/2,hi=TARGET_W-lo;auto t0=std::chrono::steady_clock::now();auto prefFull=make_row4_prefix_lut(hi,mod);auto suffFull=make_row4_suffix_lut(lo,mod);auto pref=compact_row4_lut(prefFull,hi,true);auto suff=compact_row4_lut(suffFull,lo,false);prefFull.clear();prefFull.shrink_to_fit();suffFull.clear();suffFull.shrink_to_fit();double host_s=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();Count *dpref=nullptr,*dsuff=nullptr;ck(cudaMalloc(&dpref,pref.size()*sizeof(Count)),"row4 pref");ck(cudaMalloc(&dsuff,suff.size()*sizeof(Count)),"row4 suff");ck(cudaMemcpy(dpref,pref.data(),pref.size()*sizeof(Count),cudaMemcpyHostToDevice),"row4 pref copy");ck(cudaMemcpy(dsuff,suff.data(),suff.size()*sizeof(Count),cudaMemcpyHostToDevice),"row4 suff copy");ck(cudaMemcpyToSymbol(D_R4_PREF,&dpref,sizeof(dpref)),"row4 pref ptr");ck(cudaMemcpyToSymbol(D_R4_SUFF,&dsuff,sizeof(dsuff)),"row4 suff ptr");Count*cur=nullptr;ck(cudaMalloc(&cur,size_t(outN)*sizeof(Count)),"row4 compact out");int b=int(std::min<Code>(65535,(outN+threads-1)/threads));auto g0=std::chrono::steady_clock::now();bounded_fill_row4_lut_kernel<<<std::max(1,b),threads>>>(cur,outN,W);ck(cudaDeviceSynchronize(),"row4 lut init");double gpu_s=std::chrono::duration<double>(std::chrono::steady_clock::now()-g0).count();cudaFree(dpref);cudaFree(dsuff);std::cerr<<"direct row4 lut states="<<outN<<" lut_host_s="<<host_s<<" kernel_s="<<gpu_s<<" pref_mib="<<(pref.size()*sizeof(Count)>>20)<<" suff_mib="<<(suff.size()*sizeof(Count)>>20)<<"\n";return cur;
}

static std::vector<Count> make_row3_prefix_lut(int len,Count mod){using namespace oneesan::row3auto;std::vector<Count>cur(DIM);cur[0]=1;size_t cnt=1;for(int d=0;d<len;++d){std::vector<Count>nxt(cnt*3*DIM);for(size_t c=0;c<cnt;++c)for(int a=0;a<3;++a)row3_apply_row(&cur[c*DIM],&nxt[(c*3+a)*DIM],a,mod);cur.swap(nxt);cnt*=3;}return cur;}
static std::vector<Count> make_row3_suffix_lut(int len,Count mod){using namespace oneesan::row3auto;std::vector<Count>cur(DIM);for(int i=0;i<DIM;++i)cur[i]=Count(BETA[i]%mod);size_t cnt=1,pow3=1;for(int d=0;d<len;++d){std::vector<Count>nxt(cnt*3*DIM);for(int a=0;a<3;++a)for(size_t c=0;c<cnt;++c)row3_apply_col(a,&cur[c*DIM],&nxt[(size_t(a)*pow3+c)*DIM],mod);cur.swap(nxt);cnt*=3;pow3*=3;}return cur;}
static Count* build_direct_row3_lut_compact_single_gpu(int W,Count mod,int threads,Code&outN,Code outDp[MAXW+1][MAXW+2]){
    using namespace oneesan::row3auto;cudaSetDevice(0);Code dp[MAXW+1][MAXW+2]{};build_bounded_dp(3,dp);outN=dp[W][1];std::memcpy(outDp,dp,sizeof(dp));ck(cudaMemcpyToSymbol(D_BOUND_DP,dp,sizeof(dp)),"row3 lut dp");int cap=3;ck(cudaMemcpyToSymbol(D_BOUND_CAP,&cap,sizeof(cap)),"row3 lut cap");
    constexpr int lo=TARGET_W/2,hi=TARGET_W-lo;auto t0=std::chrono::steady_clock::now();auto pref=make_row3_prefix_lut(hi,mod);auto suff=make_row3_suffix_lut(lo,mod);double host_s=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
    Count *dpref=nullptr,*dsuff=nullptr;ck(cudaMalloc(&dpref,pref.size()*sizeof(Count)),"row3 pref");ck(cudaMalloc(&dsuff,suff.size()*sizeof(Count)),"row3 suff");ck(cudaMemcpy(dpref,pref.data(),pref.size()*sizeof(Count),cudaMemcpyHostToDevice),"row3 pref copy");ck(cudaMemcpy(dsuff,suff.data(),suff.size()*sizeof(Count),cudaMemcpyHostToDevice),"row3 suff copy");ck(cudaMemcpyToSymbol(D_R3_PREF,&dpref,sizeof(dpref)),"row3 pref ptr");ck(cudaMemcpyToSymbol(D_R3_SUFF,&dsuff,sizeof(dsuff)),"row3 suff ptr");Count scale=host_pow_mod(6,W,mod),inv=host_pow_mod(scale,(unsigned long long)mod-2,mod);ck(cudaMemcpyToSymbol(D_R3_INV_SCALE,&inv,sizeof(inv)),"row3 inv scale");
    Count*cur=nullptr;ck(cudaMalloc(&cur,size_t(outN)*sizeof(Count)),"row3 compact out");int b=int(std::min<Code>(65535,(outN+threads-1)/threads));auto g0=std::chrono::steady_clock::now();bounded_fill_row3_lut_kernel<<<std::max(1,b),threads>>>(cur,outN,W);ck(cudaDeviceSynchronize(),"row3 lut init");double gpu_s=std::chrono::duration<double>(std::chrono::steady_clock::now()-g0).count();cudaFree(dpref);cudaFree(dsuff);std::cerr<<"direct row3 lut states="<<outN<<" lut_host_s="<<host_s<<" kernel_s="<<gpu_s<<" pref_mib="<<(pref.size()*sizeof(Count)>>20)<<" suff_mib="<<(suff.size()*sizeof(Count)>>20)<<"\n";return cur;
}

static void init_direct_row4_lut_inplace_single_gpu(int W,Count mod,int threads,Count*fullMain){
    using namespace oneesan::row4mod;cudaSetDevice(0);Code dp[MAXW+1][MAXW+2]{};build_bounded_dp(4,dp);Code n=dp[W][1];ck(cudaMemcpyToSymbol(D_BOUND_DP,dp,sizeof(dp)),"row4 inplace dp");int cap=4;ck(cudaMemcpyToSymbol(D_BOUND_CAP,&cap,sizeof(cap)),"row4 inplace cap");constexpr int lo=TARGET_W/2,hi=TARGET_W-lo;auto t0=std::chrono::steady_clock::now();auto prefFull=make_row4_prefix_lut(hi,mod);auto suffFull=make_row4_suffix_lut(lo,mod);auto pref=compact_row4_lut(prefFull,hi,true);auto suff=compact_row4_lut(suffFull,lo,false);double host_s=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();Count *dpref=nullptr,*dsuff=nullptr;ck(cudaMalloc(&dpref,pref.size()*sizeof(Count)),"row4 inplace pref");ck(cudaMalloc(&dsuff,suff.size()*sizeof(Count)),"row4 inplace suff");ck(cudaMemcpy(dpref,pref.data(),pref.size()*sizeof(Count),cudaMemcpyHostToDevice),"row4 inplace pref copy");ck(cudaMemcpy(dsuff,suff.data(),suff.size()*sizeof(Count),cudaMemcpyHostToDevice),"row4 inplace suff copy");ck(cudaMemcpyToSymbol(D_R4_PREF,&dpref,sizeof(dpref)),"row4 inplace pref ptr");ck(cudaMemcpyToSymbol(D_R4_SUFF,&dsuff,sizeof(dsuff)),"row4 inplace suff ptr");auto g0=std::chrono::steady_clock::now();int b=int(std::min<Code>(65535,(n+threads-1)/threads));bounded_fill_row4_lut_kernel<<<std::max(1,b),threads>>>(fullMain,n,W);ck(cudaDeviceSynchronize(),"row4 inplace init");double gpu_s=std::chrono::duration<double>(std::chrono::steady_clock::now()-g0).count();cudaFree(dpref);cudaFree(dsuff);std::cerr<<"direct row4 inplace states="<<n<<" lut_host_s="<<host_s<<" kernel_s="<<gpu_s<<"\n";bounded_expand_inplace_single_gpu(fullMain,n,W,threads);
}

static Count* build_direct_row2_bounded_compact_single_gpu(int W,int K,int threads,Code&outN,Code outDp[MAXW+1][MAXW+2]){
    if(K<2)return nullptr;
    cudaSetDevice(0); Code dpOld[MAXW+1][MAXW+2]{},dpNew[MAXW+1][MAXW+2]{};
    build_bounded_dp(2,dpOld); Code oldN=dpOld[W][1]; Count*cur=nullptr;
    ck(cudaMalloc(&cur,size_t(oldN)*sizeof(Count)),"row2 compact base");
    ck(cudaMemcpyToSymbol(D_BOUND_DP,dpOld,sizeof(dpOld)),"row2 compact dp"); int cap2=2; ck(cudaMemcpyToSymbol(D_BOUND_CAP,&cap2,sizeof(cap2)),"row2 compact cap");
    int bi=int(std::min<Code>(65535,(oldN+threads-1)/threads)); bounded_fill_row2_automaton_kernel<<<std::max(1,bi),threads>>>(cur,oldN,W); ck(cudaDeviceSynchronize(),"row2 compact init");
    std::cerr<<"direct row2 compact states="<<oldN<<"\n";
    for(int cap=3;cap<=K;++cap){
        build_bounded_dp(cap,dpNew);Code n=dpNew[W][1],dn=dpNew[W-1][1];Count *a=nullptr,*b=nullptr,*d=nullptr,*e=nullptr;
        ck(cudaMalloc(&a,size_t(n)*sizeof(Count)),"delay bounded a");ck(cudaMalloc(&b,size_t(n)*sizeof(Count)),"delay bounded b");ck(cudaMalloc(&d,size_t(dn)*sizeof(Count)),"delay bounded d");ck(cudaMalloc(&e,size_t(dn)*sizeof(Count)),"delay bounded e");
        ck(cudaMemset(a,0,size_t(n)*sizeof(Count)),"delay bounded zero a");ck(cudaMemset(d,0,size_t(dn)*sizeof(Count)),"delay bounded zero d");
        ck(cudaMemcpyToSymbol(D_BOUND_OLD_DP,dpOld,sizeof(dpOld)),"delay bounded old dp");ck(cudaMemcpyToSymbol(D_BOUND_DP,dpNew,sizeof(dpNew)),"delay bounded new dp");int oldcap=cap-1;ck(cudaMemcpyToSymbol(D_BOUND_OLD_CAP,&oldcap,sizeof(oldcap)),"delay bounded old cap");ck(cudaMemcpyToSymbol(D_BOUND_CAP,&cap,sizeof(cap)),"delay bounded cap");
        int bm=int(std::min<Code>(65535,(n+threads-1)/threads)),bd=int(std::min<Code>(65535,(dn+threads-1)/threads)),be=int(std::min<Code>(65535,(oldN+threads-1)/threads));
        bounded_embed_kernel<<<std::max(1,be),threads>>>(cur,oldN,a,W);ck(cudaDeviceSynchronize(),"delay bounded embed");cudaFree(cur);cur=a;Count*nxt=b,*dc=d,*de=e;
        for(int p=W-1;p>=1;--p){if(p>1){ck(cudaMemset(de,0,size_t(dn)*sizeof(Count)),"delay bounded clear d");bounded_reverse_main_kernel<<<std::max(1,bm),threads>>>(cur,dc,n,nxt,W,p);bounded_forward_block_kernel<<<std::max(1,bm),threads>>>(cur,n,de,W,p);}else{ck(cudaMemcpy(nxt,cur,size_t(n)*sizeof(Count),cudaMemcpyDeviceToDevice),"delay bounded identity");ck(cudaMemset(de,0,size_t(dn)*sizeof(Count)),"delay bounded clear d");bounded_main_kernel<<<std::max(1,bm),threads>>>(cur,n,nxt,de,W,p);bounded_block_kernel<<<std::max(1,bd),threads>>>(dc,dn,nxt,W,p);}ck(cudaDeviceSynchronize(),"delay bounded step");std::swap(cur,nxt);std::swap(dc,de);}
        cudaFree(nxt);cudaFree(dc);cudaFree(de);std::memcpy(dpOld,dpNew,sizeof(dpOld));oldN=n;std::cerr<<"delay bounded cap="<<cap<<" states="<<n<<" block="<<dn<<"\n";
    }
    outN=oldN;std::memcpy(outDp,dpOld,sizeof(dpOld));return cur;
}

static void run_direct_row2_bounded_prefix_single_gpu(int W,int K,Count* fullMain,int threads){
    if(K<2){run_bounded_prefix_single_gpu(W,K,fullMain,threads);return;}
    cudaSetDevice(0);
    Code dpOld[MAXW+1][MAXW+2]{},dpNew[MAXW+1][MAXW+2]{};
    build_bounded_dp(2,dpOld); Code oldN=dpOld[W][1];
    Count*cur=nullptr; ck(cudaMalloc(&cur,size_t(oldN)*sizeof(Count)),"row2 auto compact");
    ck(cudaMemcpyToSymbol(D_BOUND_DP,dpOld,sizeof(dpOld)),"row2 auto dp"); int cap2=2; ck(cudaMemcpyToSymbol(D_BOUND_CAP,&cap2,sizeof(cap2)),"row2 auto cap");
    int bi=int(std::min<Code>(65535,(oldN+threads-1)/threads));
    bounded_fill_row2_automaton_kernel<<<std::max(1,bi),threads>>>(cur,oldN,W);
    ck(cudaDeviceSynchronize(),"row2 auto init");
    std::cerr<<"direct row2 compact states="<<oldN<<"\n";
    for(int cap=3;cap<=K;++cap){
        build_bounded_dp(cap,dpNew); Code n=dpNew[W][1],dn=dpNew[W-1][1];
        Count *a=nullptr,*b=nullptr,*d=nullptr,*e=nullptr;
        ck(cudaMalloc(&a,size_t(n)*sizeof(Count)),"row2bound a"); ck(cudaMalloc(&b,size_t(n)*sizeof(Count)),"row2bound b");
        ck(cudaMalloc(&d,size_t(dn)*sizeof(Count)),"row2bound d"); ck(cudaMalloc(&e,size_t(dn)*sizeof(Count)),"row2bound e");
        ck(cudaMemset(a,0,size_t(n)*sizeof(Count)),"row2bound zero a"); ck(cudaMemset(d,0,size_t(dn)*sizeof(Count)),"row2bound zero d");
        ck(cudaMemcpyToSymbol(D_BOUND_OLD_DP,dpOld,sizeof(dpOld)),"row2bound old dp"); ck(cudaMemcpyToSymbol(D_BOUND_DP,dpNew,sizeof(dpNew)),"row2bound new dp");
        int oldcap=cap-1; ck(cudaMemcpyToSymbol(D_BOUND_OLD_CAP,&oldcap,sizeof(oldcap)),"row2bound old cap"); ck(cudaMemcpyToSymbol(D_BOUND_CAP,&cap,sizeof(cap)),"row2bound cap");
        int bm=int(std::min<Code>(65535,(n+threads-1)/threads)),bd=int(std::min<Code>(65535,(dn+threads-1)/threads)),be=int(std::min<Code>(65535,(oldN+threads-1)/threads));
        bounded_embed_kernel<<<std::max(1,be),threads>>>(cur,oldN,a,W); ck(cudaDeviceSynchronize(),"row2bound embed"); cudaFree(cur); cur=a;
        Count*nxt=b,*dc=d,*de=e;
        for(int p=W-1;p>=1;--p){
            if(p>1){ck(cudaMemset(de,0,size_t(dn)*sizeof(Count)),"row2bound clear d");bounded_reverse_main_kernel<<<std::max(1,bm),threads>>>(cur,dc,n,nxt,W,p);bounded_forward_block_kernel<<<std::max(1,bm),threads>>>(cur,n,de,W,p);}
            else {ck(cudaMemcpy(nxt,cur,size_t(n)*sizeof(Count),cudaMemcpyDeviceToDevice),"row2bound identity");ck(cudaMemset(de,0,size_t(dn)*sizeof(Count)),"row2bound clear d");bounded_main_kernel<<<std::max(1,bm),threads>>>(cur,n,nxt,de,W,p);bounded_block_kernel<<<std::max(1,bd),threads>>>(dc,dn,nxt,W,p);}
            ck(cudaDeviceSynchronize(),"row2bound step"); std::swap(cur,nxt); std::swap(dc,de);
        }
        cudaFree(nxt);cudaFree(dc);cudaFree(de);std::memcpy(dpOld,dpNew,sizeof(dpOld));oldN=n;
        std::cerr<<"direct row2 bounded cap="<<cap<<" states="<<n<<" block="<<dn<<"\n";
    }
    ck(cudaMemcpyToSymbol(D_BOUND_DP,dpOld,sizeof(dpOld)),"row2bound final dp"); ck(cudaMemcpyToSymbol(D_BOUND_CAP,&K,sizeof(K)),"row2bound final cap");
    int bs=int(std::min<Code>(65535,(oldN+threads-1)/threads)); bounded_scatter_full_kernel<<<std::max(1,bs),threads>>>(cur,oldN,W); ck(cudaDeviceSynchronize(),"row2bound scatter full"); cudaFree(cur);
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
    build_full_dp();G_FACTOR=build_factor_tables();
    int visible=0;ck(cudaGetDeviceCount(&visible),"count");
    int ng=requested<=0?visible:std::min(requested,visible);
    if(ng<1||ng>MAXGPU){std::cerr<<"need 1..8 GPUs\n";return 2;}
    int peers=0;
    for(int a=0;a<ng;++a)for(int b=0;b<ng;++b)if(a!=b){int can=0;ck(cudaDeviceCanAccessPeer(&can,a,b),"can peer");if(can){cudaSetDevice(a);auto e=cudaDeviceEnablePeerAccess(b,0);if(e==cudaErrorPeerAccessAlreadyEnabled)cudaGetLastError();else ck(e,"enable peer");peers++;}}
    if(ng>1&&peers!=ng*(ng-1)){std::cerr<<"HBM mode requires full P2P: "<<peers<<"/"<<ng*(ng-1)<<"\n";return 3;}

    std::array<uint16_t,1u<<14> trit7{};for(uint32_t c=0;c<(1u<<14);++c){uint32_t z=0,m=1;bool ok=true;for(int k=0;k<7;++k){uint32_t v=(c>>(2*k))&3u;if(v>2){ok=false;break;}z+=v*m;m*=3u;}trit7[c]=ok?uint16_t(z):uint16_t(0);}
    uint32_t *fLA[MAXGPU]{},*fLM[MAXGPU]{},*fLO[MAXGPU]{},*fLR[MAXGPU]{},*fHA[MAXGPU]{},*fHM[MAXGPU]{},*fHO[MAXGPU]{},*fHR[MAXGPU]{};uint16_t* fTrit[MAXGPU]{};Code *fHMB[MAXGPU]{},*fHBB[MAXGPU]{};
    for(int d=0;d<ng;++d){cudaSetDevice(d);ck(cudaMalloc(&fTrit[d],trit7.size()*sizeof(uint16_t)),"trit alloc");ck(cudaMemcpy(fTrit[d],trit7.data(),trit7.size()*sizeof(uint16_t),cudaMemcpyHostToDevice),"trit copy");ck(cudaMemcpyToSymbol(D_TRIT7_PTR,&fTrit[d],sizeof(fTrit[d])),"trit ptr");auto cp=[&](uint32_t**dst,const std::vector<uint32_t>&v,const char*w){if(v.empty())return;ck(cudaMalloc(dst,v.size()*sizeof(uint32_t)),w);ck(cudaMemcpy(*dst,v.data(),v.size()*sizeof(uint32_t),cudaMemcpyHostToDevice),w);};cp(&fLA[d],G_FACTOR.low_all_codes,"f low all");cp(&fLM[d],G_FACTOR.low_mask_codes,"f low mask");cp(&fLO[d],G_FACTOR.low_mask_off,"f low off");cp(&fLR[d],G_FACTOR.low_packed_rank,"f low rank");cp(&fHA[d],G_FACTOR.high_all_codes,"f high all");cp(&fHM[d],G_FACTOR.high_mask_codes,"f high mask");cp(&fHO[d],G_FACTOR.high_mask_off,"f high off");cp(&fHR[d],G_FACTOR.high_packed_rank,"f high rank");auto cpc=[&](Code**dst,const std::vector<Code>&v,const char*w){ck(cudaMalloc(dst,v.size()*sizeof(Code)),w);ck(cudaMemcpy(*dst,v.data(),v.size()*sizeof(Code),cudaMemcpyHostToDevice),w);};cpc(&fHMB[d],G_FACTOR.high_main_base,"f high main base");cpc(&fHBB[d],G_FACTOR.high_block_base,"f high block base");ck(cudaMemcpyToSymbol(D_F_LOW_ALL_CODES,&fLA[d],sizeof(fLA[d])),"f ptr");ck(cudaMemcpyToSymbol(D_F_LOW_MASK_CODES,&fLM[d],sizeof(fLM[d])),"f ptr");ck(cudaMemcpyToSymbol(D_F_LOW_MASK_OFF,&fLO[d],sizeof(fLO[d])),"f ptr");ck(cudaMemcpyToSymbol(D_F_LOW_PACKED_RANK,&fLR[d],sizeof(fLR[d])),"f ptr");ck(cudaMemcpyToSymbol(D_F_HIGH_ALL_CODES,&fHA[d],sizeof(fHA[d])),"f ptr");ck(cudaMemcpyToSymbol(D_F_HIGH_MASK_CODES,&fHM[d],sizeof(fHM[d])),"f ptr");ck(cudaMemcpyToSymbol(D_F_HIGH_MASK_OFF,&fHO[d],sizeof(fHO[d])),"f ptr");ck(cudaMemcpyToSymbol(D_F_HIGH_PACKED_RANK,&fHR[d],sizeof(fHR[d])),"f ptr");ck(cudaMemcpyToSymbol(D_F_HIGH_MAIN_BASE,&fHMB[d],sizeof(fHMB[d])),"f ptr");ck(cudaMemcpyToSymbol(D_F_HIGH_BLOCK_BASE,&fHBB[d],sizeof(fHBB[d])),"f ptr");ck(cudaMemcpyToSymbol(D_F_LOW_ALL_OFF,G_FACTOR.low_all_off.data(),sizeof(uint32_t)*(MAXW+2)),"f low all off");ck(cudaMemcpyToSymbol(D_F_HIGH_ALL_OFF,G_FACTOR.high_all_off.data(),sizeof(uint32_t)*(MAXW+2)),"f high all off");}

    Code mainN=H_DP[W][1],blockN=H_DP[W-1][1];Code mc=(mainN+ng-1)/ng,bc=(blockN+ng-1)/ng;Count*mp[MAXGPU]{},*bp[MAXGPU]{};std::vector<Code>ml(ng),bl(ng);
    for(int d=0;d<ng;++d){ml[d]=std::min<Code>(mc,mainN-std::min<Code>(mainN,Code(d)*mc));bl[d]=std::min<Code>(bc,blockN-std::min<Code>(blockN,Code(d)*bc));cudaSetDevice(d);if(ml[d])ck(cudaMalloc(&mp[d],size_t(ml[d])*sizeof(Count)),"auth main");if(bl[d])ck(cudaMalloc(&bp[d],size_t(bl[d])*sizeof(Count)),"auth block");}
    std::vector<DeviceCtx>ctx(ng);for(int d=0;d<ng;++d)ctx[d].init(d,mods[0],mp,bp,mc,bc,ng);
    size_t min_free=~size_t(0),min_total=~size_t(0);for(int d=0;d<ng;++d){ck(cudaSetDevice(d),"set meminfo");size_t f=0,t=0;ck(cudaMemGetInfo(&f,&t),"cudaMemGetInfo");min_free=std::min(min_free,f);min_total=std::min(min_total,t);}
    int reserve_mib=std::min(8192,std::max(256,int((min_total>>20)/32)));if(const char*e=std::getenv("GRIDFP_VRAM_RESERVE_MIB")){int v=std::atoi(e);if(v>=0)reserve_mib=v;}
    size_t requested_target=size_t(std::max(1,target_mib))<<20;size_t reserve=size_t(reserve_mib)<<20;if(min_free<=reserve+(64ull<<20)){std::cerr<<"insufficient HBM after authoritative state: min_free_mib="<<(min_free>>20)<<" reserve_mib="<<reserve_mib<<"\n";return 5;}
    size_t target=std::min(requested_target,min_free-reserve);int effective_target_mib=int(target>>20);
    std::cerr<<"HBM32 batch memory: auth_gib="<<double(mainN+blockN)*sizeof(Count)/(1ull<<30)<<" auth_per_gpu_gib="<<double(mainN+blockN)*sizeof(Count)/ng/(1ull<<30)<<" min_total_gib="<<double(min_total)/(1ull<<30)<<" min_free_after_auth_gib="<<double(min_free)/(1ull<<30)<<" requested_scratch_mib="<<target_mib<<" effective_scratch_mib="<<effective_target_mib<<" reserve_mib="<<reserve_mib<<" moduli="<<mods.size()<<"\n";

    if constexpr(LOW_LUT_K+HIGH_LUT_K != TARGET_W-1){std::cerr<<"forced2 requires LOW+HIGH=W-1\n";return 4;}
    int threads=256,maxgroups=0;auto prep0=std::chrono::steady_clock::now();std::vector<PreparedWindow> schedule;
    {
        const int ranges[2][2]={{W-1,LOW_LUT_K+1},{LOW_LUT_K,1}};
        for(auto const& r:ranges){
            int hi=r[0],lo=r[1];WindowPlan wp;wp.p_hi=hi;wp.p_lo=lo;wp.fixed_pos=window_candidates(W,hi,lo);
            int k=(int)wp.fixed_pos.size();int nj=1<<k;PreparedWindow pw;pw.wp=wp;pw.groups.reserve(nj);size_t mx=0;
            for(int g=0;g<nj;++g){auto pg=prepare_group(W,pw.wp,g,mc,bc,ng);size_t b=size_t(2*pg.ms.size+2*pg.ds.size)*sizeof(Count);mx=std::max(mx,b);pw.groups.push_back(std::move(pg));}
            if(mx>target){std::cerr<<"forced window does not fit p="<<hi<<".."<<lo<<" max_bytes="<<mx<<" target="<<target<<"\n";return 4;}
            maxgroups=std::max(maxgroups,nj);std::sort(pw.groups.begin(),pw.groups.end(),[](auto const&a,auto const&b){return a.work>b.work;});
            std::cerr<<"forced window p="<<hi<<".."<<lo<<" fixed="<<k<<" groups="<<nj<<" max_mib="<<(mx>>20)<<"\n";schedule.push_back(std::move(pw));
        }
    }
    double prepare_s=std::chrono::duration<double>(std::chrono::steady_clock::now()-prep0).count();std::cerr<<"prepared windows="<<schedule.size()<<" max_groups="<<maxgroups<<" prepare_s="<<prepare_s<<"\n";

    MateID init=MateID(R)<<(2*(W-1));Code ig=rank_full(init,W);int io=int(ig/mc);Code fg=rank_full(MateID(R),W);int fo=int(fg/mc);Count one=1;
    for(size_t ri=0;ri<mods.size();++ri){Count mod=mods[ri];
        for(int d=0;d<ng;++d){ck(cudaSetDevice(d),"set residue reset");ck(cudaMemcpyToSymbol(D_MOD,&mod,sizeof(mod)),"set modulus");if(ml[d])ck(cudaMemset(mp[d],0,size_t(ml[d])*sizeof(Count)),"zero main");if(bl[d])ck(cudaMemset(bp[d],0,size_t(bl[d])*sizeof(Count)),"zero block");ck(cudaDeviceSynchronize(),"zero sync");ctx[d].active=0;ctx[d].groups=0;}
        auto wall0=std::chrono::steady_clock::now();
        int prefixK=5;if(const char*e=std::getenv("GRIDFP_BOUNDED_PREFIX_K"))prefixK=std::max(1,std::min(W-2,std::atoi(e)));
        bool requestDirect5=(prefixK==5);if(const char*e=std::getenv("GRIDFP_DIRECT_ROW5_LUT"))requestDirect5=std::atoi(e)!=0;
        bool requestDirect6=(prefixK==6);if(const char*e=std::getenv("GRIDFP_DIRECT_ROW6_LUT"))requestDirect6=std::atoi(e)!=0;
        if(ng!=1&&prefixK>1&&!((requestDirect5&&prefixK==5)||(requestDirect6&&prefixK==6))){std::cerr<<"bounded prefix prototype currently requires NGPU=1 except direct row5/6\n";return 31;}
        if(prefixK==1){
            for(int d=0;d<ng;++d){ck(cudaSetDevice(d),"set row1 init device");Code total=Code(1)<<(W-1);Code mine=(total+ng-1-d)/ng;int blocks=int(std::min<Code>(65535,(mine+threads-1)/threads));if(mine)init_after_first_row_kernel<<<std::max(1,blocks),threads>>>(d,ng);}for(int d=0;d<ng;++d){ck(cudaSetDevice(d),"row1 init sync set");ck(cudaDeviceSynchronize(),"row1 init sync");}
        }else {
            bool direct2=true;if(const char*e=std::getenv("GRIDFP_DIRECT_ROW2"))direct2=std::atoi(e)!=0;
            bool direct5=requestDirect5;
            bool direct6=requestDirect6;
            bool inplace4=(prefixK==4);if(const char*e=std::getenv("GRIDFP_INPLACE_ROW4"))inplace4=std::atoi(e)!=0;
            bool delayAuth=false;if(const char*e=std::getenv("GRIDFP_DELAY_AUTH"))delayAuth=std::atoi(e)!=0;
            if(direct6&&prefixK==6){
                init_direct_row6_lut_full_multi_gpu(W,mod,threads,mp,ml,mc,ng);
            }else if(direct5&&prefixK==5){
                init_direct_row5_lut_full_multi_gpu(W,mod,threads,mp,ml,mc,ng);
            }else if(inplace4&&prefixK==4&&ng==1){
                init_direct_row4_lut_inplace_single_gpu(W,mod,threads,mp[0]);
            }else if(delayAuth&&direct2&&ng==1){
                ck(cudaSetDevice(0),"delay auth set");if(mp[0]){cudaFree(mp[0]);mp[0]=nullptr;}if(bp[0]){cudaFree(bp[0]);bp[0]=nullptr;}
                Code compactN=0,compactDp[MAXW+1][MAXW+2]{};bool direct4=(prefixK==4);if(const char*e=std::getenv("GRIDFP_DIRECT_ROW4_LUT"))direct4=std::atoi(e)!=0;bool direct3=(prefixK==3);if(const char*e=std::getenv("GRIDFP_DIRECT_ROW3_LUT"))direct3=std::atoi(e)!=0;Count*compact=direct4&&prefixK==4?build_direct_row4_lut_compact_single_gpu(W,mod,threads,compactN,compactDp):(direct3&&prefixK==3?build_direct_row3_lut_compact_single_gpu(W,mod,threads,compactN,compactDp):build_direct_row2_bounded_compact_single_gpu(W,prefixK,threads,compactN,compactDp));
                ck(cudaMalloc(&mp[0],size_t(ml[0])*sizeof(Count)),"delay auth main alloc");ck(cudaMemset(mp[0],0,size_t(ml[0])*sizeof(Count)),"delay auth main zero");
                ck(cudaMemcpyToSymbol(D_MAIN_PTR,mp,sizeof(Count*)*MAXGPU),"delay auth main ptr");ck(cudaMemcpyToSymbol(D_BOUND_DP,compactDp,sizeof(compactDp)),"delay auth scatter dp");ck(cudaMemcpyToSymbol(D_BOUND_CAP,&prefixK,sizeof(prefixK)),"delay auth scatter cap");
                int bs=int(std::min<Code>(65535,(compactN+threads-1)/threads));bounded_scatter_full_kernel<<<std::max(1,bs),threads>>>(compact,compactN,W);ck(cudaDeviceSynchronize(),"delay auth scatter");cudaFree(compact);
                ck(cudaMalloc(&bp[0],size_t(bl[0])*sizeof(Count)),"delay auth block alloc");ck(cudaMemset(bp[0],0,size_t(bl[0])*sizeof(Count)),"delay auth block zero");ck(cudaMemcpyToSymbol(D_BLOCK_PTR,bp,sizeof(Count*)*MAXGPU),"delay auth block ptr");
                size_t ff=0,tt=0;ck(cudaMemGetInfo(&ff,&tt),"delay auth meminfo");std::cerr<<"delay auth restored full arrays free_mib="<<(ff>>20)<<"\n";
            }else if(direct2)run_direct_row2_bounded_prefix_single_gpu(W,prefixK,mp[0],threads);else run_bounded_prefix_single_gpu(W,prefixK,mp[0],threads);
        }
        int done_windows=0;
        for(int row=prefixK;row<W-1;++row){for(auto const&pw:schedule){int nj=(int)pw.groups.size();std::atomic<int>next{0};std::vector<std::thread>ths;ths.reserve(ng);for(int d=0;d<ng;++d)ths.emplace_back([&,d]{for(;;){int q=next.fetch_add(1,std::memory_order_relaxed);if(q>=nj)break;process_group(ctx[d],W,pw.wp,pw.groups[q],threads,target);}});for(auto&t:ths)t.join();++done_windows;}std::cerr<<"mod "<<(ri+1)<<"/"<<mods.size()<<" p="<<mod<<" row "<<row+1<<"/"<<W<<"\n";}
        Count ans=0;
        for(int d=0;d<ng;++d){
            ck(cudaSetDevice(d),"final sum set");Count* da=nullptr;ck(cudaMalloc(&da,sizeof(Count)),"final sum malloc");ck(cudaMemset(da,0,sizeof(Count)),"final sum zero");
            Code total=Code(1)<<(W-1);Code mine=(total+ng-1-d)/ng;int blocks=int(std::min<Code>(65535,(mine+threads-1)/threads));
            if(mine)sum_first_row_states_kernel<<<std::max(1,blocks),threads,threads*sizeof(unsigned long long)>>>(d,ng,da);
            Count x=0;ck(cudaMemcpy(&x,da,sizeof(x),cudaMemcpyDeviceToHost),"final sum copy");cudaFree(da);ans=(ans>=mod-x)?ans-(mod-x):ans+x;
        }
        double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-wall0).count();double mx=0,sum=0;size_t maxIntervals=0;for(auto&c:ctx){mx=std::max(mx,c.active);sum+=c.active;maxIntervals=std::max(maxIntervals,c.maxIntervals);}std::cout<<"backend=gridfp-b300-hbm32-factorized-batch n="<<n<<" residue="<<ans<<" modulus="<<mod<<" residue_index="<<ri<<" residues_total="<<mods.size()<<" gpus="<<ng<<" peers="<<peers<<" main_states="<<mainN<<" blocked_states="<<blockN<<" scratch_target_mib="<<effective_target_mib<<" windows="<<done_windows<<" max_groups="<<maxgroups<<" max_intervals="<<maxIntervals<<" active_max_s="<<mx<<" active_sum_s="<<sum<<" prepare_s="<<prepare_s<<" wall_s="<<wall<<std::endl;
    }

    for(auto&c:ctx)c.destroy();for(int d=0;d<ng;++d){cudaSetDevice(d);if(mp[d])cudaFree(mp[d]);if(bp[d])cudaFree(bp[d]);if(fLA[d])cudaFree(fLA[d]);if(fLM[d])cudaFree(fLM[d]);if(fLO[d])cudaFree(fLO[d]);if(fLR[d])cudaFree(fLR[d]);if(fHA[d])cudaFree(fHA[d]);if(fHM[d])cudaFree(fHM[d]);if(fHO[d])cudaFree(fHO[d]);if(fHR[d])cudaFree(fHR[d]);if(fHMB[d])cudaFree(fHMB[d]);if(fHBB[d])cudaFree(fHBB[d]);if(fTrit[d])cudaFree(fTrit[d]);}
}
