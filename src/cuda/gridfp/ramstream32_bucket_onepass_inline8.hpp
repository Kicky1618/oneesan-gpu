#pragma once

#include "ramstream32_bucket_fused.cuh"
#include "ramstream32_bucket_reverse_fused.cuh"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

// 64-bit one-pass closure record:
//   bits  0..27 : begin in the residual source stream
//   bits 28..31 : ordinary count (0..15)
//   bits 32..35 : CROSS count (0..15)
//   bits 36..63 : one inline 28-bit source descriptor
//
// local source descriptors use 24 bits; CROSS uses 28 bits. If ordinary
// sources exist, inline the first ordinary source. Otherwise inline the first
// CROSS source. The type is therefore implied by (local_count != 0).
using BucketOnePassInline8Rec = uint64_t;
static constexpr uint32_t OP8I_BEGIN_BITS=28;
static constexpr uint32_t OP8I_BEGIN_MASK=(1u<<OP8I_BEGIN_BITS)-1u;
static constexpr uint32_t OP8I_SRC_MASK=(1u<<28)-1u;
static constexpr int OP8I_LC_SHIFT=28;
static constexpr int OP8I_CC_SHIFT=32;
static constexpr int OP8I_SRC_SHIFT=36;

#if defined(__CUDACC__)
#define OP8I_HD __host__ __device__ __forceinline__
#else
#define OP8I_HD inline
#endif
OP8I_HD uint32_t op8i_begin(BucketOnePassInline8Rec r){return uint32_t(r)&OP8I_BEGIN_MASK;}
OP8I_HD uint32_t op8i_lc(BucketOnePassInline8Rec r){return uint32_t((r>>OP8I_LC_SHIFT)&0xfu);}
OP8I_HD uint32_t op8i_cc(BucketOnePassInline8Rec r){return uint32_t((r>>OP8I_CC_SHIFT)&0xfu);}
OP8I_HD uint32_t op8i_inline(BucketOnePassInline8Rec r){return uint32_t((r>>OP8I_SRC_SHIFT)&OP8I_SRC_MASK);}
#undef OP8I_HD

struct BucketOnePassInline8SideHost {
    std::vector<BucketOnePassInline8Rec> rec;
    std::vector<uint32_t> src;
    size_t bytes()const{return rec.size()*sizeof(BucketOnePassInline8Rec)+src.size()*sizeof(uint32_t);}
};
struct BucketForwardOnePassInline8Host {
    BucketOnePassInline8SideHost low,high;
    size_t bytes()const{return low.bytes()+high.bytes();}
};
struct BucketReverseOnePassInline8Host {
    BucketOnePassInline8SideHost low,high;
    size_t bytes()const{return low.bytes()+high.bytes();}
};

static void bucket_onepass_inline8_append(
    BucketOnePassInline8SideHost&out,const BucketFusedDst&r,
    const std::vector<uint32_t>&local,const std::vector<uint32_t>&cross,const char*what
){
    uint32_t lc=r.counts&0xffffu,cc=r.counts>>16;
    if(lc>15||cc>15||lc+cc==0){std::cerr<<"inline8 count unsupported "<<what<<" local="<<lc<<" cross="<<cc<<'\n';std::exit(470);}
    if(uint64_t(r.local_begin)+lc>local.size()||uint64_t(r.cross_begin)+cc>cross.size()){std::cerr<<"inline8 source slice overflow "<<what<<'\n';std::exit(471);}
    if(out.src.size()>OP8I_BEGIN_MASK){std::cerr<<"inline8 residual source stream exceeds 28-bit begin "<<what<<" entries="<<out.src.size()<<'\n';std::exit(472);}
    uint32_t inl=lc?local[r.local_begin]:cross[r.cross_begin];
    if(inl&~OP8I_SRC_MASK){std::cerr<<"inline8 source descriptor overflow "<<what<<" value="<<inl<<'\n';std::exit(473);}
    uint32_t begin=uint32_t(out.src.size());
    for(uint32_t i=lc?1u:0u;i<lc;++i)out.src.push_back(local[r.local_begin+i]);
    for(uint32_t i=lc?0u:1u;i<cc;++i)out.src.push_back(cross[r.cross_begin+i]);
    BucketOnePassInline8Rec w=uint64_t(begin)|(uint64_t(lc)<<OP8I_LC_SHIFT)|(uint64_t(cc)<<OP8I_CC_SHIFT)|(uint64_t(inl)<<OP8I_SRC_SHIFT);
    out.rec.push_back(w);
}

static BucketForwardOnePassInline8Host build_bucket_forward_onepass_inline8(const BucketFusedHost&f){
    BucketForwardOnePassInline8Host out;out.low.rec.reserve(f.low_dst.size());out.high.rec.reserve(f.high_dst.size());
    size_t le=f.low_local_src.size()+f.low_cross_op.size(),he=f.high_local_src.size()+f.high_cross_op.size();
    if(le<f.low_dst.size()||he<f.high_dst.size()){std::cerr<<"inline8 forward records exceed sources\n";std::exit(474);}
    out.low.src.reserve(le-f.low_dst.size());out.high.src.reserve(he-f.high_dst.size());
    for(const auto&r:f.low_dst)bucket_onepass_inline8_append(out.low,r,f.low_local_src,f.low_cross_op,"forward-low");
    for(const auto&r:f.high_dst)bucket_onepass_inline8_append(out.high,r,f.high_local_src,f.high_cross_op,"forward-high");
    if(out.low.src.size()!=le-f.low_dst.size()||out.high.src.size()!=he-f.high_dst.size())std::exit(475);
    size_t old=(f.low_dst.size()+f.high_dst.size())*sizeof(BucketFusedDst)+(le+he)*sizeof(uint32_t);
    std::cerr<<"bucket_forward_onepass_inline8 records="<<(out.low.rec.size()+out.high.rec.size())<<" residual_sources="<<(out.low.src.size()+out.high.src.size())<<" old_mib="<<double(old)/double(1<<20)<<" inline8_mib="<<double(out.bytes())/double(1<<20)<<"\n";return out;
}
static BucketReverseOnePassInline8Host build_bucket_reverse_onepass_inline8(const ReverseBucketFusedHost&f){
    BucketReverseOnePassInline8Host out;out.low.rec.reserve(f.low_dst.size());out.high.rec.reserve(f.high_dst.size());
    size_t le=f.low_local_src.size()+f.low_cross_op.size(),he=f.high_local_src.size()+f.high_cross_op.size();
    if(le<f.low_dst.size()||he<f.high_dst.size()){std::cerr<<"inline8 reverse records exceed sources\n";std::exit(476);}
    out.low.src.reserve(le-f.low_dst.size());out.high.src.reserve(he-f.high_dst.size());
    for(const auto&r:f.low_dst)bucket_onepass_inline8_append(out.low,r,f.low_local_src,f.low_cross_op,"reverse-low");
    for(const auto&r:f.high_dst)bucket_onepass_inline8_append(out.high,r,f.high_local_src,f.high_cross_op,"reverse-high");
    if(out.low.src.size()!=le-f.low_dst.size()||out.high.src.size()!=he-f.high_dst.size())std::exit(477);
    size_t old=(f.low_dst.size()+f.high_dst.size())*sizeof(BucketFusedDst)+(le+he)*sizeof(uint32_t);
    std::cerr<<"bucket_reverse_onepass_inline8 records="<<(out.low.rec.size()+out.high.rec.size())<<" residual_sources="<<(out.low.src.size()+out.high.src.size())<<" old_mib="<<double(old)/double(1<<20)<<" inline8_mib="<<double(out.bytes())/double(1<<20)<<"\n";return out;
}
