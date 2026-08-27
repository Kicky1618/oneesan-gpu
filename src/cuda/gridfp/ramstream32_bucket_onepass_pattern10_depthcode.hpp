#pragma once

#include "ramstream32_bucket_onepass_zero_alias.cuh"
#include "ramstream32_bucket_closure_pattern10_depth8.hpp"
#include "ramstream32_bucket_reverse_split54.cuh"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

// Re-encode (pattern10, CROSS depth4) into the already-free upper 10 bits of
// each 64-bit orbit word.  A host-only all-legal-factor-state proof
// (gridfp_pattern10_depthcode_bound.cpp) establishes that W=28 needs at most
// 718 pairs in any (phase,side,p,height) context, so phase mode is sufficient
// without first materializing a byte/nibble depth sidecar.
enum P10DepthCodeMode : uint32_t { P10DC_COARSE=0, P10DC_PHASE=1, P10DC_STREAM=2 };
static constexpr uint32_t P10DC_STREAMS=4;
static constexpr uint32_t P10DC_HDIM=TARGET_W+1;
static constexpr uint32_t P10DC_KEY_COUNT=2u*2u*P10DC_STREAMS*uint32_t(TARGET_W)*P10DC_HDIM;
static constexpr uint32_t P10DC_INVALID_BASE=0xffffffffu;
static constexpr uint32_t P10DC_PAIR_COUNT=1u<<14;
static constexpr uint32_t P10DC_WORDS=P10DC_PAIR_COUNT/64u;

#if defined(__CUDACC__)
#define P10DC_HD __host__ __device__ __forceinline__
#else
#define P10DC_HD inline
#endif
P10DC_HD uint32_t p10dc_key(bool rev,bool high,uint32_t sid,int p,uint32_t h,uint32_t mode){
    uint32_t r=mode>=P10DC_PHASE?uint32_t(rev):0u;
    uint32_t s=mode>=P10DC_STREAM?(sid&3u):0u;
    return ((((r*2u+uint32_t(high))*P10DC_STREAMS+s)*uint32_t(TARGET_W)+uint32_t(p))*P10DC_HDIM+h);
}
#undef P10DC_HD

struct P10DepthCodeBits {
    std::array<uint64_t,P10DC_WORDS> bits{};
    uint16_t count=0;
    bool add(uint16_t pair){uint32_t w=pair>>6,b=pair&63u;uint64_t m=1ull<<b;if(bits[w]&m)return false;bits[w]|=m;++count;return true;}
};

struct P10DepthCodeBookHost {
    uint32_t mode=P10DC_PHASE;
    std::vector<uint32_t> base;
    std::vector<uint16_t> decode;
    size_t bytes()const{return base.size()*sizeof(uint32_t)+decode.size()*sizeof(uint16_t);}
};
struct BucketForwardPattern10DepthCodeHost { size_t bytes()const{return 0;} };
struct BucketReversePattern10DepthCodeHost {
    ReverseSplit54Host split;
    P10DepthCodeBookHost codebook;
    size_t bytes()const{return split.bytes()+codebook.bytes();}
};

static BucketForwardPattern10DepthCodeHost build_bucket_forward_pattern10_depthcode_placeholder(
    const StorageLayout&,BucketOrbitStreamsHost&,BucketFusedHost&
){return {};}

static uint16_t p10dc_low_pair_host(MateID d,int p,bool active,const char*what){
    using namespace oneesan::gridfp;
    if(!active)return uint16_t(CLOSURE_PATTERN10_NONE<<4);
    uint16_t id=closure_pattern10_encode(d,LOW_LUT_K+1,p);MateID source=0;int dep=low_cross_preimage_partial(d,LOW_LUT_K+1,p,source);
    if(dep<0||dep>15){std::cerr<<"depthcode LOW overflow "<<what<<" p="<<p<<" depth="<<dep<<'\n';std::exit(591);}if(id==CLOSURE_PATTERN10_NONE)dep=0;return uint16_t((id<<4)|uint16_t(dep));
}
static uint16_t p10dc_high_pair_host(MateID d,int rel,bool active,const char*what){
    using namespace oneesan::gridfp;
    if(!active)return uint16_t(CLOSURE_PATTERN10_NONE<<4);
    uint16_t id=closure_pattern10_encode(d,HIGH_LUT_K+1,rel);MateID source=0;int dep=high_cross_preimage_partial(d,HIGH_LUT_K+1,rel,source);
    if(dep<0||dep>15){std::cerr<<"depthcode HIGH overflow "<<what<<" rel="<<rel<<" depth="<<dep<<'\n';std::exit(592);}if(id==CLOSURE_PATTERN10_NONE)dep=0;return uint16_t((id<<4)|uint16_t(dep));
}

struct P10DepthCodeEntryView {
    BucketOrbitOp* op=nullptr;
    uint16_t pair=0;
    bool rev=false,high=false;
    uint32_t sid=0;
    int p=0;
    uint32_t h=0;
};

template<class F>
static void p10dc_for_each_entry_direct(
    const StorageLayout&layout,BucketOrbitStreamsHost&bo,ReverseSplit54Host&rs,
    const BucketFusedHost&bf,F&&fn
){
    size_t lp=size_t(bo.low_nblocks)+1,hp=size_t(bo.high_nblocks)+1;
    for(int p=LOW_LUT_K;p>=1;--p){uint32_t pi=uint32_t(LOW_LUT_K-p);for(uint32_t bid=0;bid<bo.low_nblocks;++bid){const auto&xb=layout.main_blocks[bid];if(!xb.valid)continue;const auto&db=p==1?xb:layout.block_blocks[xb.he];
        auto scan=[&](auto&ops,const auto&off,uint32_t sid,bool active,const char*what){for(uint32_t q=off[size_t(pi)*lp+bid];q<off[size_t(pi)*lp+bid+1];++q){uint16_t pair=uint16_t(oneesan::gridfp::CLOSURE_PATTERN10_NONE<<4);if(active){uint32_t loc=p==1?bkf_orbit_src(ops[q]):bkf_orbit_drop(ops[q]),dc=bkcp10_low_code_host(bf,loc,db.hs);MateID d=p==1?(MateID(dc)|(MateID(db.c)<<(2*LOW_LUT_K))):minsert(MateID(dc),p,N);pair=p10dc_low_pair_host(d,p,true,what);}fn(P10DepthCodeEntryView{&ops[q],pair,false,false,sid,p,uint32_t(xb.he)});}};
        scan(bo.low_nn,bo.low_nn_off,0,true,"forward-low-nn");scan(bo.low_nr,bo.low_nr_off,1,p!=1,"forward-low-nr");scan(bo.low_nl,bo.low_nl_off,2,p!=1,"forward-low-nl");
    }}
    for(int p=TARGET_W-1;p>=LOW_LUT_K+1;--p){uint32_t pi=uint32_t((TARGET_W-1)-p);int rel=p-LOW_LUT_K;for(uint32_t bid=0;bid<bo.high_nblocks;++bid){const auto&xb=layout.main_blocks[bid];if(!xb.valid)continue;const auto&db=layout.block_blocks[xb.hs];
        auto scan=[&](auto&ops,const auto&off,uint32_t sid,const char*what){for(uint32_t q=off[size_t(pi)*hp+bid];q<off[size_t(pi)*hp+bid+1];++q){uint32_t loc=bkf_orbit_drop(ops[q]),dc=bkcp10_high_code_host(bf,loc,db.he);MateID d=minsert(MateID(dc),rel,N);uint16_t pair=p10dc_high_pair_host(d,rel,true,what);fn(P10DepthCodeEntryView{&ops[q],pair,false,true,sid,p,uint32_t(xb.hs)});}};
        scan(bo.high_nn,bo.high_nn_off,0,"forward-high-nn");scan(bo.high_nrnl,bo.high_nrnl_off,3,"forward-high-nrnl");
    }}
    size_t rp=size_t(rs.nblocks)+1;
    for(int p=1;p<=LOW_LUT_K;++p){uint32_t pi=uint32_t(p-1);for(uint32_t bid=0;bid<rs.nblocks;++bid){const auto&xb=layout.main_blocks[bid];if(!xb.valid)continue;const auto&db=layout.block_blocks[xb.he];
        auto scan=[&](auto&ops,const auto&off,uint32_t sid,const char*what){for(uint32_t q=off[size_t(pi)*rp+bid];q<off[size_t(pi)*rp+bid+1];++q){uint32_t loc=bkf_orbit_drop(ops[q]),dc=bkcp10_low_code_host(bf,loc,db.hs);MateID d=blocked_exclude_reverse(MateID(dc),LOW_LUT_K+1,p);uint16_t pair=p10dc_low_pair_host(d,p,true,what);fn(P10DepthCodeEntryView{&ops[q],pair,true,false,sid,p,uint32_t(xb.he)});}};
        scan(rs.low.nn,rs.low.nn_off,0,"reverse-low-nn");scan(rs.low.nr,rs.low.nr_off,1,"reverse-low-nr");scan(rs.low.nl,rs.low.nl_off,2,"reverse-low-nl");
    }}
    for(int p=LOW_LUT_K+1;p<TARGET_W;++p){uint32_t pi=uint32_t(p-(LOW_LUT_K+1));int rel=p-LOW_LUT_K;bool edge=p==TARGET_W-1;for(uint32_t bid=0;bid<rs.nblocks;++bid){const auto&xb=layout.main_blocks[bid];if(!xb.valid)continue;const auto&db=edge?xb:layout.block_blocks[xb.hs];
        auto scan=[&](auto&ops,const auto&off,uint32_t sid,bool nn,const char*what){for(uint32_t q=off[size_t(pi)*rp+bid];q<off[size_t(pi)*rp+bid+1];++q){bool active=!edge||nn;uint16_t pair=uint16_t(oneesan::gridfp::CLOSURE_PATTERN10_NONE<<4);if(active){uint32_t loc=edge?bkf_orbit_src(ops[q]):bkf_orbit_drop(ops[q]),dc=bkcp10_high_code_host(bf,loc,db.he);MateID d=edge?(MateID(db.c)|(MateID(dc)<<2)):blocked_exclude_reverse(MateID(dc),HIGH_LUT_K+1,rel);pair=p10dc_high_pair_host(d,rel,true,what);}fn(P10DepthCodeEntryView{&ops[q],pair,true,true,sid,p,uint32_t(xb.hs)});}};
        scan(rs.high.nn,rs.high.nn_off,0,true,"reverse-high-nn");scan(rs.high.nr,rs.high.nr_off,1,false,"reverse-high-nr");scan(rs.high.nl,rs.high.nl_off,2,false,"reverse-high-nl");
    }}
}

static P10DepthCodeBookHost p10dc_build_and_rewrite_direct(
    const StorageLayout&layout,BucketOrbitStreamsHost&bo,ReverseSplit54Host&rs,
    const BucketFusedHost&bf
){
    constexpr uint32_t mode=P10DC_PHASE;
    std::vector<P10DepthCodeBits> ctx(P10DC_KEY_COUNT);
    uint64_t visited=0;
    p10dc_for_each_entry_direct(layout,bo,rs,bf,[&](P10DepthCodeEntryView e){uint32_t k=p10dc_key(e.rev,e.high,e.sid,e.p,e.h,mode);if(k>=ctx.size()){std::cerr<<"depthcode key overflow k="<<k<<'\n';std::exit(593);}ctx[k].add(e.pair);++visited;});
    uint32_t max_pairs=0;size_t used_contexts=0,total_pairs=0;for(const auto&c:ctx)if(c.count){++used_contexts;total_pairs+=c.count;max_pairs=std::max<uint32_t>(max_pairs,c.count);}if(max_pairs>1024){std::cerr<<"phase depthcode bound violated max_pairs="<<max_pairs<<"; run gridfp_pattern10_depthcode_bound probe\n";std::exit(594);}

    P10DepthCodeBookHost out;out.mode=mode;out.base.assign(P10DC_KEY_COUNT,P10DC_INVALID_BASE);out.decode.reserve(total_pairs);
    for(uint32_t k=0;k<ctx.size();++k){const auto&c=ctx[k];if(!c.count)continue;out.base[k]=uint32_t(out.decode.size());for(uint32_t w=0;w<P10DC_WORDS;++w){uint64_t x=c.bits[w];while(x){uint32_t b=uint32_t(__builtin_ctzll(x));out.decode.push_back(uint16_t((w<<6)+b));x&=x-1;}}if(out.decode.size()-out.base[k]!=c.count)std::exit(595);}

    uint64_t rewritten=0;
    p10dc_for_each_entry_direct(layout,bo,rs,bf,[&](P10DepthCodeEntryView e){uint32_t k=p10dc_key(e.rev,e.high,e.sid,e.p,e.h,mode),b=out.base[k];if(b==P10DC_INVALID_BASE)std::exit(596);uint32_t n=ctx[k].count;auto first=out.decode.begin()+b,last=first+n,it=std::lower_bound(first,last,e.pair);if(it==last||*it!=e.pair)std::exit(597);uint32_t code=uint32_t(it-first);if(code>1023)std::exit(598);*e.op=bkcp10_set(*e.op,uint16_t(code));if(out.decode[b+code]!=e.pair)std::exit(599);++rewritten;});
    if(rewritten!=visited)std::exit(600);
    std::cerr<<"pattern10_depthcode direct_build=1 selected_mode="<<mode<<" contexts="<<used_contexts<<" total_pairs="<<total_pairs<<" max_pairs="<<max_pairs<<" rewritten_ops="<<rewritten<<" codebook_mib="<<double(out.bytes())/double(1<<20)<<" sidecar_bytes_per_orbit=0 temporary_depth_bytes=0\n";
    return out;
}

static BucketReversePattern10DepthCodeHost build_bucket_reverse_pattern10_depthcode_zero_checked(
    const StorageLayout&layout,BucketOrbitStreamsHost&bo,BucketFusedHost&bf,
    ReverseBucketAtomicHost&rb,ReverseBucketFusedHost&rf
){
    BucketReversePattern10DepthCodeHost out;out.split=build_reverse_split54(layout,rb,true);
    out.codebook=p10dc_build_and_rewrite_direct(layout,bo,out.split,bf);
    bucket_zero_release_forward_closure(bf);bucket_zero_release_reverse_closure(rf);
    return out;
}
