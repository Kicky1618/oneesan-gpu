#pragma once

#include "ramstream32_bucket_closure_zero.cuh"

#ifndef BKCZ_TERNARY_KEY4
#define BKCZ_TERNARY_KEY4 1
#endif
static_assert(BKCZ_TERNARY_KEY4==0||BKCZ_TERNARY_KEY4==1,"BKCZ_TERNARY_KEY4 must be 0 or 1");

// LL/RR remote-candidate positions form no-adjacent masks on each side of the
// NN pair. For an active factor of K symbols plus center, ordinary sources are
// one implicit RL source plus at most ceil(K/2) remote LL/RR sources. This is
// an exact topology bound, not an empirical W=28 observation. At K<=14 the
// per-orbit plan therefore needs at most 8 local descriptors instead of K.
static constexpr int BKCZ_MAX_FACTOR=(LOW_LUT_K>HIGH_LUT_K?LOW_LUT_K:HIGH_LUT_K);
static constexpr int BKCZ_MAX_LOCAL=1+(BKCZ_MAX_FACTOR+1)/2;
static_assert(BKCZ_MAX_LOCAL>0,"zero-closure source plan requires non-empty factors");
static_assert(BKCZ_MAX_LOCAL<=255,"zero-closure local_n no longer fits uint8_t");
static_assert(BKCZ_MAX_FACTOR<=14||BKCZ_MAX_LOCAL<=16,"unexpected closure plan growth");
struct BkczPlan{
    uint32_t local[BKCZ_MAX_LOCAL];
    uint32_t cross_src=0;
    uint8_t local_n=0,cross_depth=0,cross_valid=0,pad=0;
};

__device__ __forceinline__ void bkcz_plan_add_low(BkczPlan&p,MateID x,int fixed_he){uint32_t loc=0,bid=0;if(bkcz_low_source_ref(x,fixed_he,loc,bid)){if(p.local_n>=BKCZ_MAX_LOCAL)return;p.local[p.local_n++]=bkf_src_pack(bid,loc);}}
__device__ __forceinline__ void bkcz_plan_add_high(BkczPlan&p,MateID x,int fixed_hs){uint32_t loc=0,bid=0;if(bkcz_high_source_ref(x,fixed_hs,loc,bid)){if(p.local_n>=BKCZ_MAX_LOCAL)return;p.local[p.local_n++]=bkf_src_pack(bid,loc);}}

__device__ __forceinline__ BkczPlan bkcz_build_low_plan(MateID d,int fixed_he,int p){
    BkczPlan z{};if(mpair(d,p)!=NN)return z;MateID x=msetpair(d,p,RL);bkcz_plan_add_low(z,x,fixed_he);
    int bal=0;for(int q=p-2;q>=0;--q){MateValue v=mget(d,q);if(bal==0&&v==L){x=msetpair(d,p,LL);x=mset(x,q,R);bkcz_plan_add_low(z,x,fixed_he);}if(v==L)++bal;else if(v==R)--bal;if(bal<0)break;}
    bal=0;bool cross_ok=true;for(int q=p+1;q<LOW_LUT_K+1;++q){MateValue v=mget(d,q);if(bal==0&&v==R){x=msetpair(d,p,RR);x=mset(x,q,L);bkcz_plan_add_low(z,x,fixed_he);}if(v==R)++bal;else if(v==L)--bal;if(bal<0){cross_ok=false;break;}}
    if(cross_ok){int depth=bal+1;x=msetpair(d,p,RR);uint32_t loc=0,bid=0;if(bkcz_low_source_ref(x,fixed_he+2,loc,bid)){z.cross_src=bkf_src_pack(bid,loc);z.cross_depth=uint8_t(depth);z.cross_valid=1;}}return z;
}
__device__ __forceinline__ BkczPlan bkcz_build_high_plan(MateID d,int fixed_hs,int rel){
    BkczPlan z{};if(mpair(d,rel)!=NN)return z;MateID x=msetpair(d,rel,RL);bkcz_plan_add_high(z,x,fixed_hs);
    int bal=0;for(int q=rel+1;q<HIGH_LUT_K+1;++q){MateValue v=mget(d,q);if(bal==0&&v==R){x=msetpair(d,rel,RR);x=mset(x,q,L);bkcz_plan_add_high(z,x,fixed_hs);}if(v==R)++bal;else if(v==L)--bal;if(bal<0)break;}
    bal=0;bool cross_ok=true;for(int q=rel-2;q>=0;--q){MateValue v=mget(d,q);if(bal==0&&v==L){x=msetpair(d,rel,LL);x=mset(x,q,R);bkcz_plan_add_high(z,x,fixed_hs);}if(v==L)++bal;else if(v==R)--bal;if(bal<0){cross_ok=false;break;}}
    if(cross_ok){int depth=bal+1;x=msetpair(d,rel,LL);uint32_t loc=0,bid=0;if(bkcz_high_source_ref(x,fixed_hs+2,loc,bid)){z.cross_src=bkf_src_pack(bid,loc);z.cross_depth=uint8_t(depth);z.cross_valid=1;}}return z;
}

__device__ __forceinline__ BkczPlan bkcz_forward_low_plan(uint32_t dest_loc,const BucketPhysicalBlock&db,int p){uint32_t dc=bkci_low_code(dest_loc,db.hs);MateID d=p==1?(MateID(dc)|(MateID(db.c)<<(2*LOW_LUT_K))):minsert(MateID(dc),p,N);return bkcz_build_low_plan(d,db.he,p);}
__device__ __forceinline__ BkczPlan bkcz_forward_high_plan(uint32_t dest_loc,const BucketPhysicalBlock&db,int p){uint32_t dc=bkci_high_code(dest_loc,db.he);int rel=p-LOW_LUT_K;MateID d=minsert(MateID(dc),rel,N);return bkcz_build_high_plan(d,db.hs,rel);}
__device__ __forceinline__ BkczPlan bkcz_reverse_low_plan(uint32_t dest_loc,const BucketPhysicalBlock&db,int p){uint32_t dc=bkci_low_code(dest_loc,db.hs);MateID d=blocked_exclude_reverse(MateID(dc),LOW_LUT_K+1,p);return bkcz_build_low_plan(d,db.he,p);}
__device__ __forceinline__ BkczPlan bkcz_reverse_high_plan(uint32_t dest_loc,const BucketPhysicalBlock&db,int p,bool edge){uint32_t dc=bkci_high_code(dest_loc,db.he);int rel=p-LOW_LUT_K;MateID d=edge?(MateID(db.c)|(MateID(dc)<<2)):blocked_exclude_reverse(MateID(dc),HIGH_LUT_K+1,rel);return bkcz_build_high_plan(d,db.hs,rel);}

static constexpr uint32_t bkcz_pow3_const(int n){return n<=0?1u:3u*bkcz_pow3_const(n-1);}

// Factor codes contain only N/R/L, packed as base-4 digits 0/1/2. Convert four
// symbols at a time: a 4-symbol chunk contributes d0+3*d1+9*d2+27*d3, and
// successive chunks are separated by 3^4=81. W=28 therefore needs only four
// unrolled chunk iterations for either 13/14-symbol factor instead of 13/14
// serial multiply-add steps in gdx_ternary_key(). Bits above K are zero in all
// StorageFactorHost codes, so the final partial chunk needs no mask.
template<int K>
__host__ __device__ __forceinline__ uint32_t bkcz_ternary_key4_legal(uint32_t code){
    static_assert(K>0&&K<=16,"chunked ternary key expects a 32-bit legal factor code");
    uint32_t key=0,weight=1u;
#pragma unroll
    for(int pos=0;pos<K;pos+=4){
        uint32_t x=code>>(2*pos);
        uint32_t chunk=(x&3u)+3u*((x>>2)&3u)+9u*((x>>4)&3u)+27u*((x>>6)&3u);
        key+=chunk*weight;
        weight*=81u;
    }
    return key;
}

template<int K>
__device__ __forceinline__ uint32_t bkcz_ternary_key(uint32_t code){
#if BKCZ_TERNARY_KEY4
    return bkcz_ternary_key4_legal<K>(code);
#else
    return gdx_ternary_key<K>(code);
#endif
}

#if GPU_DIRECT_PM_ACCUM
using BkczCrossAccum=uint64_t;
__device__ __forceinline__ BkczCrossAccum bkcz_cross_add(BkczCrossAccum a,Count b){return a+uint64_t(b);}
#else
using BkczCrossAccum=Count;
__device__ __forceinline__ BkczCrossAccum bkcz_cross_add(BkczCrossAccum a,Count b){return gpu_direct_add(a,b);}
#endif

// Changing one ternary digit R(1)->L(2) adds 3^pos to the direct-table key;
// L(2)->R(1) subtracts it. Occupancy is unchanged, so a valid preimage remains
// on D_BKF_FIXED_OWNER. The flip also changes the factor boundary height by
// exactly +2, which is the CROSS source-height invariant checked by
// ramstream32_cross_height_delta_plan.cu. Therefore a valid direct-table entry
// already has the required owner and height; only its owner-local rank is used.
__device__ __forceinline__ BkczCrossAccum bkcz_sum_high_preimages_fast(
    uint32_t dest_code,uint32_t depth,uint32_t source_bid,uint32_t source_low_loc
){
    BkczCrossAccum sum=0;int s=int(depth);uint32_t low_slot=bkf_loc_owner(source_low_loc),low_rank=bkf_loc_rank(source_low_loc);BucketPhysicalBlock sb=bkf_low_main(low_slot,source_bid);
    uint32_t key=bkcz_ternary_key<HIGH_LUT_K>(dest_code),weight=1u;
#pragma unroll
    for(int pos=0;pos<HIGH_LUT_K;++pos){
        uint32_t v=(dest_code>>(2*pos))&3u;
        if(v==uint32_t(::L)){if(s==1)break;--s;}
        else if(v==uint32_t(R)){
            if(s==1){uint32_t x=D_BKF_HIGH_DIRECT[key+weight];if(x!=BKF_DIRECT_INVALID){uint32_t hr=bkf_loc_rank(x);sum=bkcz_cross_add(sum,bkf_ptr(low_slot,sb.off+Code(hr)*sb.cols+low_rank)[0]);}}
            ++s;
        }
        weight*=3u;
    }
    return sum;
}

__device__ __forceinline__ BkczCrossAccum bkcz_sum_low_preimages_fast(
    uint32_t dest_code,uint32_t depth,uint32_t source_bid,uint32_t source_high_loc
){
    BkczCrossAccum sum=0;int s=int(depth);uint32_t high_slot=bkf_loc_owner(source_high_loc),high_rank=bkf_loc_rank(source_high_loc);BucketPhysicalBlock sb=bkf_high_main(high_slot,source_bid);
    uint32_t key=bkcz_ternary_key<LOW_LUT_K>(dest_code),weight=bkcz_pow3_const(LOW_LUT_K-1);
#pragma unroll
    for(int pos=LOW_LUT_K-1;pos>=0;--pos){
        uint32_t v=(dest_code>>(2*pos))&3u;
        if(v==uint32_t(R)){if(s==1)break;--s;}
        else if(v==uint32_t(::L)){
            if(s==1){uint32_t x=D_BKF_LOW_DIRECT[key-weight];if(x!=BKF_DIRECT_INVALID){uint32_t lr=bkf_loc_rank(x);sum=bkcz_cross_add(sum,bkf_ptr(high_slot,sb.off+Code(high_rank)*sb.cols+lr)[0]);}}
            ++s;
        }
        if(pos)weight/=3u;
    }
    return sum;
}

__device__ __forceinline__ Count bkcz_low_plan_sum(const BkczPlan&p,const BucketPhysicalBlock&db,uint32_t hr){
#if GPU_DIRECT_PM_ACCUM
    uint64_t sum=0;
#else
    Count sum=0;
#endif
    for(uint32_t i=0;i<p.local_n;++i){uint32_t x=p.local[i],sl=bkf_src_locator(x),ss=bkf_loc_owner(sl);BucketPhysicalBlock sb=bkf_low_main(ss,bkf_src_block(x));Count v=bkf_ptr(ss,sb.off+Code(hr)*sb.cols+bkf_loc_rank(sl))[0];
#if GPU_DIRECT_PM_ACCUM
        sum+=uint64_t(v);
#else
        sum=gpu_direct_add(sum,v);
#endif
    }
    if(p.cross_valid){uint32_t x=p.cross_src,sl=bkf_src_locator(x);uint32_t dc=D_BKF_HIGH_CODES[D_BKF_HIGH_CODE_OFF[size_t(D_BKF_FIXED_OWNER)*D_BKF_CODE_PITCH+db.he]+hr];
#if GPU_DIRECT_PM_ACCUM
        sum+=bkcz_sum_high_preimages_fast(dc,p.cross_depth,bkf_src_block(x),sl);
#else
        sum=gpu_direct_add(sum,bkcz_sum_high_preimages_fast(dc,p.cross_depth,bkf_src_block(x),sl));
#endif
    }
#if GPU_DIRECT_PM_ACCUM
    return gpu_direct_pm_reduce_u64(sum);
#else
    return sum;
#endif
}

__device__ __forceinline__ Count bkcz_high_plan_sum(const BkczPlan&p,const BucketPhysicalBlock&db,uint32_t lr){
#if GPU_DIRECT_PM_ACCUM
    uint64_t sum=0;
#else
    Count sum=0;
#endif
    for(uint32_t i=0;i<p.local_n;++i){uint32_t x=p.local[i],sl=bkf_src_locator(x),ss=bkf_loc_owner(sl);BucketPhysicalBlock sb=bkf_high_main(ss,bkf_src_block(x));Count v=bkf_ptr(ss,sb.off+Code(bkf_loc_rank(sl))*sb.cols+lr)[0];
#if GPU_DIRECT_PM_ACCUM
        sum+=uint64_t(v);
#else
        sum=gpu_direct_add(sum,v);
#endif
    }
    if(p.cross_valid){uint32_t x=p.cross_src,sl=bkf_src_locator(x);uint32_t dc=D_BKF_LOW_CODES[D_BKF_LOW_CODE_OFF[size_t(D_BKF_FIXED_OWNER)*D_BKF_CODE_PITCH+db.hs]+lr];
#if GPU_DIRECT_PM_ACCUM
        sum+=bkcz_sum_low_preimages_fast(dc,p.cross_depth,bkf_src_block(x),sl);
#else
        sum=gpu_direct_add(sum,bkcz_sum_low_preimages_fast(dc,p.cross_depth,bkf_src_block(x),sl));
#endif
    }
#if GPU_DIRECT_PM_ACCUM
    return gpu_direct_pm_reduce_u64(sum);
#else
    return sum;
#endif
}
