#pragma once

#include "../../common/gridfp_closure_inverse.hpp"
#include "ramstream32_bucket_fused.cuh"
#include "ramstream32_bucket_reverse_fused.cuh"
#ifndef GPU_DIRECT_PM_ACCUM
#define GPU_DIRECT_PM_ACCUM 0
#endif
#if GPU_DIRECT_PM_ACCUM
#include "ramstream32_bucket_fused_pm.cuh"
#endif

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

// Ordinary LL/RR/RL closure sources are reconstructed algorithmically from the
// destination active-half code.  Only CROSS metadata remains resident.
// Record ids and per-(step,destination-block) offsets stay identical to the
// existing BucketFusedHost / ReverseBucketFusedHost ordering, so packed18 and
// split18 attachment metadata can be reused unchanged.
//
// One 32-bit record per closure destination:
//   bits 0..27  : begin in CROSS op array
//   bits 28..31 : CROSS op count (0..15)
using BucketClosureInverseRec = uint32_t;
static constexpr uint32_t BKCI_BEGIN_MASK=(1u<<28)-1u;
static constexpr int BKCI_COUNT_SHIFT=28;

#if defined(__CUDACC__)
#define BKCI_HD __host__ __device__ __forceinline__
#else
#define BKCI_HD inline
#endif
BKCI_HD uint32_t bkci_begin(BucketClosureInverseRec r){return r&BKCI_BEGIN_MASK;}
BKCI_HD uint32_t bkci_count(BucketClosureInverseRec r){return r>>BKCI_COUNT_SHIFT;}
#undef BKCI_HD

struct BucketClosureInverseSideHost {
    std::vector<BucketClosureInverseRec> rec;
    std::vector<uint32_t> cross;
    size_t bytes()const{return (rec.size()+cross.size())*sizeof(uint32_t);}
};
struct BucketForwardClosureInverseHost {
    BucketClosureInverseSideHost low,high;
    size_t bytes()const{return low.bytes()+high.bytes();}
};
struct BucketReverseClosureInverseHost {
    BucketClosureInverseSideHost low,high;
    size_t bytes()const{return low.bytes()+high.bytes();}
};

static BucketClosureInverseRec bkci_pack(uint32_t begin,uint32_t count,const char*what){
    if(begin>BKCI_BEGIN_MASK||count>15){
        std::cerr<<"closure inverse record overflow "<<what<<" begin="<<begin<<" count="<<count<<'\n';
        std::exit(520);
    }
    return begin|(count<<BKCI_COUNT_SHIFT);
}
static BucketClosureInverseSideHost bkci_build_side(
    const std::vector<BucketFusedDst>&dst,const std::vector<uint32_t>&cross,const char*what
){
    BucketClosureInverseSideHost out;out.rec.reserve(dst.size());out.cross=cross;
    for(const auto&r:dst){uint32_t cc=r.counts>>16;out.rec.push_back(bkci_pack(r.cross_begin,cc,what));}
    return out;
}
static BucketForwardClosureInverseHost build_bucket_forward_closure_inverse(const BucketFusedHost&f){
    BucketForwardClosureInverseHost out;out.low=bkci_build_side(f.low_dst,f.low_cross_op,"forward-low");out.high=bkci_build_side(f.high_dst,f.high_cross_op,"forward-high");
    size_t old=(f.low_dst.size()+f.high_dst.size())*sizeof(BucketFusedDst)+(f.low_local_src.size()+f.high_local_src.size()+f.low_cross_op.size()+f.high_cross_op.size())*sizeof(uint32_t);
    std::cerr<<"bucket_forward_closure_inverse records="<<(out.low.rec.size()+out.high.rec.size())<<" cross_entries="<<(out.low.cross.size()+out.high.cross.size())<<" old_mib="<<double(old)/double(1<<20)<<" inverse_mib="<<double(out.bytes())/double(1<<20)<<" ordinary_source_entries=0\n";
    return out;
}
static BucketReverseClosureInverseHost build_bucket_reverse_closure_inverse(const ReverseBucketFusedHost&f){
    BucketReverseClosureInverseHost out;out.low=bkci_build_side(f.low_dst,f.low_cross_op,"reverse-low");out.high=bkci_build_side(f.high_dst,f.high_cross_op,"reverse-high");
    size_t old=(f.low_dst.size()+f.high_dst.size())*sizeof(BucketFusedDst)+(f.low_local_src.size()+f.high_local_src.size()+f.low_cross_op.size()+f.high_cross_op.size())*sizeof(uint32_t);
    std::cerr<<"bucket_reverse_closure_inverse records="<<(out.low.rec.size()+out.high.rec.size())<<" cross_entries="<<(out.low.cross.size()+out.high.cross.size())<<" old_mib="<<double(old)/double(1<<20)<<" inverse_mib="<<double(out.bytes())/double(1<<20)<<" ordinary_source_entries=0\n";
    return out;
}

__constant__ BucketClosureInverseRec *D_BKCI_F_LOW_REC,*D_BKCI_F_HIGH_REC,*D_BKCI_R_LOW_REC,*D_BKCI_R_HIGH_REC;
__constant__ uint32_t *D_BKCI_F_LOW_CROSS,*D_BKCI_F_HIGH_CROSS,*D_BKCI_R_LOW_CROSS,*D_BKCI_R_HIGH_CROSS;

struct BucketForwardClosureInverseDeviceTables {
    BucketClosureInverseRec *low_rec=nullptr,*high_rec=nullptr;uint32_t *low_cross=nullptr,*high_cross=nullptr;
    template<class T>static void cp(T*&d,const std::vector<T>&s,const char*w){if(s.empty())return;ck(cudaMalloc(&d,s.size()*sizeof(T)),w);ck(cudaMemcpy(d,s.data(),s.size()*sizeof(T),cudaMemcpyHostToDevice),w);}
    void install(const BucketForwardClosureInverseHost&h){cp(low_rec,h.low.rec,"bkci f low rec");cp(high_rec,h.high.rec,"bkci f high rec");cp(low_cross,h.low.cross,"bkci f low cross");cp(high_cross,h.high.cross,"bkci f high cross");ck(cudaMemcpyToSymbol(D_BKCI_F_LOW_REC,&low_rec,sizeof(low_rec)),"bkci f low rec ptr");ck(cudaMemcpyToSymbol(D_BKCI_F_HIGH_REC,&high_rec,sizeof(high_rec)),"bkci f high rec ptr");ck(cudaMemcpyToSymbol(D_BKCI_F_LOW_CROSS,&low_cross,sizeof(low_cross)),"bkci f low cross ptr");ck(cudaMemcpyToSymbol(D_BKCI_F_HIGH_CROSS,&high_cross,sizeof(high_cross)),"bkci f high cross ptr");}
    void release(){if(low_rec)cudaFree(low_rec);if(high_rec)cudaFree(high_rec);if(low_cross)cudaFree(low_cross);if(high_cross)cudaFree(high_cross);low_rec=high_rec=nullptr;low_cross=high_cross=nullptr;}
};
struct BucketReverseClosureInverseDeviceTables {
    BucketClosureInverseRec *low_rec=nullptr,*high_rec=nullptr;uint32_t *low_cross=nullptr,*high_cross=nullptr;
    template<class T>static void cp(T*&d,const std::vector<T>&s,const char*w){if(s.empty())return;ck(cudaMalloc(&d,s.size()*sizeof(T)),w);ck(cudaMemcpy(d,s.data(),s.size()*sizeof(T),cudaMemcpyHostToDevice),w);}
    void install(const BucketReverseClosureInverseHost&h){cp(low_rec,h.low.rec,"bkci r low rec");cp(high_rec,h.high.rec,"bkci r high rec");cp(low_cross,h.low.cross,"bkci r low cross");cp(high_cross,h.high.cross,"bkci r high cross");ck(cudaMemcpyToSymbol(D_BKCI_R_LOW_REC,&low_rec,sizeof(low_rec)),"bkci r low rec ptr");ck(cudaMemcpyToSymbol(D_BKCI_R_HIGH_REC,&high_rec,sizeof(high_rec)),"bkci r high rec ptr");ck(cudaMemcpyToSymbol(D_BKCI_R_LOW_CROSS,&low_cross,sizeof(low_cross)),"bkci r low cross ptr");ck(cudaMemcpyToSymbol(D_BKCI_R_HIGH_CROSS,&high_cross,sizeof(high_cross)),"bkci r high cross ptr");}
    void release(){if(low_rec)cudaFree(low_rec);if(high_rec)cudaFree(high_rec);if(low_cross)cudaFree(low_cross);if(high_cross)cudaFree(high_cross);low_rec=high_rec=nullptr;low_cross=high_cross=nullptr;}
};

__device__ __forceinline__ int bkci_delta(uint32_t c){return c==uint32_t(::L)?1:(c==uint32_t(R)?-1:0);}
__device__ __forceinline__ uint32_t bkci_low_code(uint32_t loc,int h){uint32_t g=bkf_loc_owner(loc),r=bkf_loc_rank(loc);return D_BKF_LOW_CODES[D_BKF_LOW_CODE_OFF[size_t(g)*D_BKF_CODE_PITCH+h]+r];}
__device__ __forceinline__ uint32_t bkci_high_code(uint32_t loc,int h){uint32_t g=bkf_loc_owner(loc),r=bkf_loc_rank(loc);return D_BKF_HIGH_CODES[D_BKF_HIGH_CODE_OFF[size_t(g)*D_BKF_CODE_PITCH+h]+r];}

__device__ __forceinline__ Count bkci_low_source_value(MateID partial,int fixed_he,uint32_t hr){
    constexpr uint64_t MASK=(uint64_t(1)<<(2*LOW_LUT_K))-1ull;uint32_t c=uint32_t(mget(partial,LOW_LUT_K)),lc=uint32_t(partial&MASK);int hs=fixed_he+bkci_delta(c);if(hs<0||hs>LOW_LUT_K+1)return 0;
    uint32_t z=D_BKF_LOW_DIRECT[gdx_ternary_key<LOW_LUT_K>(lc)];if(z==BKF_DIRECT_INVALID||int(bkf_direct_height(z))!=hs)return 0;uint32_t sl=bkf_direct_locator(z),bid=uint32_t(3*fixed_he+int(c)),ss=bkf_loc_owner(sl);if(bid>=D_BKF_MAIN_NBLOCKS)return 0;BucketPhysicalBlock sb=bkf_low_main(ss,bid);if(!sb.valid||hr>=sb.rows)return 0;return bkf_ptr(ss,sb.off+Code(hr)*sb.cols+bkf_loc_rank(sl))[0];
}
__device__ __forceinline__ Count bkci_high_source_value(MateID partial,int fixed_hs,uint32_t lr){
    constexpr uint64_t MASK=(uint64_t(1)<<(2*HIGH_LUT_K))-1ull;uint32_t c=uint32_t(mget(partial,0)),hc=uint32_t((partial>>2)&MASK);int he=fixed_hs-bkci_delta(c);if(he<0||he>HIGH_LUT_K+1)return 0;
    uint32_t z=D_BKF_HIGH_DIRECT[gdx_ternary_key<HIGH_LUT_K>(hc)];if(z==BKF_DIRECT_INVALID||int(bkf_direct_height(z))!=he)return 0;uint32_t sl=bkf_direct_locator(z),bid=uint32_t(3*he+int(c)),ss=bkf_loc_owner(sl);if(bid>=D_BKF_MAIN_NBLOCKS)return 0;BucketPhysicalBlock sb=bkf_high_main(ss,bid);if(!sb.valid||lr>=sb.cols)return 0;return bkf_ptr(ss,sb.off+Code(bkf_loc_rank(sl))*sb.cols+lr)[0];
}
__device__ __forceinline__ uint64_t bkci_low_source_value_u64(MateID partial,int fixed_he,uint32_t hr){return uint64_t(bkci_low_source_value(partial,fixed_he,hr));}
__device__ __forceinline__ uint64_t bkci_high_source_value_u64(MateID partial,int fixed_hs,uint32_t lr){return uint64_t(bkci_high_source_value(partial,fixed_hs,lr));}

__device__ __forceinline__ Count bkci_forward_low_local(uint32_t dest_loc,const BucketPhysicalBlock&db,uint32_t hr,int p){
    uint32_t dc=bkci_low_code(dest_loc,db.hs);MateID partial=p==1?(MateID(dc)|(MateID(db.c)<<(2*LOW_LUT_K))):minsert(MateID(dc),p,N);MateID cand[32]{};int n=ordinary_closure_preimages_partial(partial,LOW_LUT_K+1,p,cand);Count sum=0;for(int i=0;i<n;++i)sum=gpu_direct_add(sum,bkci_low_source_value(cand[i],db.he,hr));return sum;
}
__device__ __forceinline__ Count bkci_forward_high_local(uint32_t dest_loc,const BucketPhysicalBlock&db,uint32_t lr,int p){
    uint32_t dc=bkci_high_code(dest_loc,db.he);int rel=p-LOW_LUT_K;MateID partial=minsert(MateID(dc),rel,N);MateID cand[32]{};int n=ordinary_closure_preimages_partial(partial,HIGH_LUT_K+1,rel,cand);Count sum=0;for(int i=0;i<n;++i)sum=gpu_direct_add(sum,bkci_high_source_value(cand[i],db.hs,lr));return sum;
}
__device__ __forceinline__ Count bkci_reverse_low_local(uint32_t dest_loc,const BucketPhysicalBlock&db,uint32_t hr,int p){
    uint32_t dc=bkci_low_code(dest_loc,db.hs);MateID partial=blocked_exclude_reverse(MateID(dc),LOW_LUT_K+1,p);MateID cand[32]{};int n=ordinary_closure_preimages_partial_reverse(partial,LOW_LUT_K+1,p,cand);Count sum=0;for(int i=0;i<n;++i)sum=gpu_direct_add(sum,bkci_low_source_value(cand[i],db.he,hr));return sum;
}
__device__ __forceinline__ Count bkci_reverse_high_local(uint32_t dest_loc,const BucketPhysicalBlock&db,uint32_t lr,int p,bool edge){
    uint32_t dc=bkci_high_code(dest_loc,db.he);int rel=p-LOW_LUT_K;MateID partial=edge?(MateID(db.c)|(MateID(dc)<<2)):blocked_exclude_reverse(MateID(dc),HIGH_LUT_K+1,rel);MateID cand[32]{};int n=ordinary_closure_preimages_partial_reverse(partial,HIGH_LUT_K+1,rel,cand);Count sum=0;for(int i=0;i<n;++i)sum=gpu_direct_add(sum,bkci_high_source_value(cand[i],db.hs,lr));return sum;
}

#if GPU_DIRECT_PM_ACCUM
__device__ __forceinline__ uint64_t bkci_forward_low_local_u64(uint32_t dest_loc,const BucketPhysicalBlock&db,uint32_t hr,int p){uint32_t dc=bkci_low_code(dest_loc,db.hs);MateID partial=p==1?(MateID(dc)|(MateID(db.c)<<(2*LOW_LUT_K))):minsert(MateID(dc),p,N);MateID cand[32]{};int n=ordinary_closure_preimages_partial(partial,LOW_LUT_K+1,p,cand);uint64_t sum=0;for(int i=0;i<n;++i)sum+=bkci_low_source_value_u64(cand[i],db.he,hr);return sum;}
__device__ __forceinline__ uint64_t bkci_forward_high_local_u64(uint32_t dest_loc,const BucketPhysicalBlock&db,uint32_t lr,int p){uint32_t dc=bkci_high_code(dest_loc,db.he);int rel=p-LOW_LUT_K;MateID partial=minsert(MateID(dc),rel,N);MateID cand[32]{};int n=ordinary_closure_preimages_partial(partial,HIGH_LUT_K+1,rel,cand);uint64_t sum=0;for(int i=0;i<n;++i)sum+=bkci_high_source_value_u64(cand[i],db.hs,lr);return sum;}
__device__ __forceinline__ uint64_t bkci_reverse_low_local_u64(uint32_t dest_loc,const BucketPhysicalBlock&db,uint32_t hr,int p){uint32_t dc=bkci_low_code(dest_loc,db.hs);MateID partial=blocked_exclude_reverse(MateID(dc),LOW_LUT_K+1,p);MateID cand[32]{};int n=ordinary_closure_preimages_partial_reverse(partial,LOW_LUT_K+1,p,cand);uint64_t sum=0;for(int i=0;i<n;++i)sum+=bkci_low_source_value_u64(cand[i],db.he,hr);return sum;}
__device__ __forceinline__ uint64_t bkci_reverse_high_local_u64(uint32_t dest_loc,const BucketPhysicalBlock&db,uint32_t lr,int p,bool edge){uint32_t dc=bkci_high_code(dest_loc,db.he);int rel=p-LOW_LUT_K;MateID partial=edge?(MateID(db.c)|(MateID(dc)<<2)):blocked_exclude_reverse(MateID(dc),HIGH_LUT_K+1,rel);MateID cand[32]{};int n=ordinary_closure_preimages_partial_reverse(partial,HIGH_LUT_K+1,rel,cand);uint64_t sum=0;for(int i=0;i<n;++i)sum+=bkci_high_source_value_u64(cand[i],db.hs,lr);return sum;}
#endif
