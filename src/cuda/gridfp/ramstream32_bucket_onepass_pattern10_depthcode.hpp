#pragma once

#include "ramstream32_bucket_onepass_zero_alias.cuh"
#include "ramstream32_bucket_closure_pattern10_depth8.hpp"
#include "ramstream32_bucket_reverse_split54.cuh"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

// Re-encode (pattern10, CROSS depth4) into the already-free upper 10 bits of
// each 64-bit orbit word.  The code is interpreted in a small topology
// context.  We try the coarsest context first and refine only when more than
// 1024 distinct pairs occur:
//   mode 0: (side,p,height)
//   mode 1: + snake phase (forward/reverse)
//   mode 2: + orbit stream (NN/NR/NL/NRNL)
// Device decode is context_base + code10 -> 14-bit {pattern10,depth4}.
// There is no per-orbit sidecar.
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
    uint32_t mode=P10DC_STREAM;
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

struct P10DepthCodeEntryView {
    BucketOrbitOp* op=nullptr;
    uint8_t depth=0;
    bool rev=false,high=false;
    uint32_t sid=0;
    int p=0;
    uint32_t h=0;
};

template<class F>
static void p10dc_for_each_entry(
    const StorageLayout&layout,BucketOrbitStreamsHost&bo,ReverseSplit54Host&rs,
    BucketPattern10Depth8Host&d,F&&fn
){
    size_t lp=size_t(bo.low_nblocks)+1,hp=size_t(bo.high_nblocks)+1;
    for(int p=LOW_LUT_K;p>=1;--p){uint32_t pi=uint32_t(LOW_LUT_K-p);for(uint32_t bid=0;bid<bo.low_nblocks;++bid){const auto&xb=layout.main_blocks[bid];if(!xb.valid)continue;
        auto scan=[&](auto&ops,const auto&off,auto&dep,uint32_t sid){for(uint32_t q=off[size_t(pi)*lp+bid];q<off[size_t(pi)*lp+bid+1];++q)fn(P10DepthCodeEntryView{&ops[q],dep[q],false,false,sid,p,uint32_t(xb.he)});};
        scan(bo.low_nn,bo.low_nn_off,d.f_low_nn,0);scan(bo.low_nr,bo.low_nr_off,d.f_low_nr,1);scan(bo.low_nl,bo.low_nl_off,d.f_low_nl,2);
    }}
    for(int p=TARGET_W-1;p>=LOW_LUT_K+1;--p){uint32_t pi=uint32_t((TARGET_W-1)-p);for(uint32_t bid=0;bid<bo.high_nblocks;++bid){const auto&xb=layout.main_blocks[bid];if(!xb.valid)continue;
        auto scan=[&](auto&ops,const auto&off,auto&dep,uint32_t sid){for(uint32_t q=off[size_t(pi)*hp+bid];q<off[size_t(pi)*hp+bid+1];++q)fn(P10DepthCodeEntryView{&ops[q],dep[q],false,true,sid,p,uint32_t(xb.hs)});};
        scan(bo.high_nn,bo.high_nn_off,d.f_high_nn,0);scan(bo.high_nrnl,bo.high_nrnl_off,d.f_high_nrnl,3);
    }}
    size_t rp=size_t(rs.nblocks)+1;
    for(int p=1;p<=LOW_LUT_K;++p){uint32_t pi=uint32_t(p-1);for(uint32_t bid=0;bid<rs.nblocks;++bid){const auto&xb=layout.main_blocks[bid];if(!xb.valid)continue;
        auto scan=[&](auto&ops,const auto&off,auto&dep,uint32_t sid){for(uint32_t q=off[size_t(pi)*rp+bid];q<off[size_t(pi)*rp+bid+1];++q)fn(P10DepthCodeEntryView{&ops[q],dep[q],true,false,sid,p,uint32_t(xb.he)});};
        scan(rs.low.nn,rs.low.nn_off,d.r_low_nn,0);scan(rs.low.nr,rs.low.nr_off,d.r_low_nr,1);scan(rs.low.nl,rs.low.nl_off,d.r_low_nl,2);
    }}
    for(int p=LOW_LUT_K+1;p<TARGET_W;++p){uint32_t pi=uint32_t(p-(LOW_LUT_K+1));for(uint32_t bid=0;bid<rs.nblocks;++bid){const auto&xb=layout.main_blocks[bid];if(!xb.valid)continue;
        auto scan=[&](auto&ops,const auto&off,auto&dep,uint32_t sid){for(uint32_t q=off[size_t(pi)*rp+bid];q<off[size_t(pi)*rp+bid+1];++q)fn(P10DepthCodeEntryView{&ops[q],dep[q],true,true,sid,p,uint32_t(xb.hs)});};
        scan(rs.high.nn,rs.high.nn_off,d.r_high_nn,0);scan(rs.high.nr,rs.high.nr_off,d.r_high_nr,1);scan(rs.high.nl,rs.high.nl_off,d.r_high_nl,2);
    }}
}

static P10DepthCodeBookHost p10dc_build_and_rewrite(
    const StorageLayout&layout,BucketOrbitStreamsHost&bo,ReverseSplit54Host&rs,
    BucketPattern10Depth8Host&depths
){
    std::vector<P10DepthCodeBits> ctx(P10DC_KEY_COUNT);
    uint32_t chosen=99,max_pairs=0;size_t used_contexts=0,total_pairs=0;
    for(uint32_t mode=P10DC_COARSE;mode<=P10DC_STREAM;++mode){
        std::fill(ctx.begin(),ctx.end(),P10DepthCodeBits{});max_pairs=0;used_contexts=0;total_pairs=0;
        p10dc_for_each_entry(layout,bo,rs,depths,[&](P10DepthCodeEntryView e){
            uint16_t pat=bkcp10_id(*e.op);uint16_t pair=uint16_t((pat<<4)|(e.depth&15u));uint32_t k=p10dc_key(e.rev,e.high,e.sid,e.p,e.h,mode);if(k>=ctx.size()){std::cerr<<"depthcode key overflow k="<<k<<'\n';std::exit(591);}ctx[k].add(pair);
        });
        for(const auto&c:ctx)if(c.count){++used_contexts;total_pairs+=c.count;max_pairs=std::max<uint32_t>(max_pairs,c.count);}
        std::cerr<<"pattern10_depthcode try_mode="<<mode<<" contexts="<<used_contexts<<" total_pairs="<<total_pairs<<" max_pairs="<<max_pairs<<" fits10="<<(max_pairs<=1024?1:0)<<'\n';
        if(max_pairs<=1024){chosen=mode;break;}
    }
    if(chosen> P10DC_STREAM){std::cerr<<"pattern10 depthcode needs >1024 pairs even with phase+stream context; keep depth4 sidecar\n";std::exit(592);}

    P10DepthCodeBookHost out;out.mode=chosen;out.base.assign(P10DC_KEY_COUNT,P10DC_INVALID_BASE);out.decode.reserve(total_pairs);
    for(uint32_t k=0;k<ctx.size();++k){const auto&c=ctx[k];if(!c.count)continue;out.base[k]=uint32_t(out.decode.size());for(uint32_t w=0;w<P10DC_WORDS;++w){uint64_t x=c.bits[w];while(x){uint32_t b=uint32_t(__builtin_ctzll(x));out.decode.push_back(uint16_t((w<<6)+b));x&=x-1;}}if(out.decode.size()-out.base[k]!=c.count)std::exit(593);}

    uint64_t rewritten=0;
    p10dc_for_each_entry(layout,bo,rs,depths,[&](P10DepthCodeEntryView e){
        uint16_t pat=bkcp10_id(*e.op);uint16_t pair=uint16_t((pat<<4)|(e.depth&15u));uint32_t k=p10dc_key(e.rev,e.high,e.sid,e.p,e.h,chosen);uint32_t b=out.base[k];if(b==P10DC_INVALID_BASE)std::exit(594);uint32_t n=ctx[k].count;auto first=out.decode.begin()+b,last=first+n,it=std::lower_bound(first,last,pair);if(it==last||*it!=pair)std::exit(595);uint32_t code=uint32_t(it-first);if(code>1023)std::exit(596);*e.op=bkcp10_set(*e.op,uint16_t(code));if(out.decode[b+code]!=pair)std::exit(597);++rewritten;
    });
    std::cerr<<"pattern10_depthcode selected_mode="<<chosen<<" contexts="<<used_contexts<<" total_pairs="<<total_pairs<<" max_pairs="<<max_pairs<<" rewritten_ops="<<rewritten<<" codebook_mib="<<double(out.bytes())/double(1<<20)<<" sidecar_bytes_per_orbit=0\n";
    return out;
}

static BucketReversePattern10DepthCodeHost build_bucket_reverse_pattern10_depthcode_zero_checked(
    const StorageLayout&layout,BucketOrbitStreamsHost&bo,BucketFusedHost&bf,
    ReverseBucketAtomicHost&rb,ReverseBucketFusedHost&rf
){
    BucketReversePattern10DepthCodeHost out;out.split=build_reverse_split54(layout,rb,true);
    build_bucket_forward_pattern10(layout,bo,bf);build_reverse_split54_pattern10(layout,bf,out.split);
    BucketPattern10Depth8Host depths=build_bucket_pattern10_depth8(layout,bf,bo,out.split);
    out.codebook=p10dc_build_and_rewrite(layout,bo,out.split,depths);
    bucket_zero_release_forward_closure(bf);bucket_zero_release_reverse_closure(rf);
    return out;
}
