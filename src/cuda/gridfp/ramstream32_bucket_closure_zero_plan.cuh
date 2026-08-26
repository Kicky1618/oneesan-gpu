#pragma once

#include "ramstream32_bucket_closure_zero.cuh"

static constexpr int BKCZ_MAX_LOCAL=16;
// One RL predecessor plus at most every other position in the same factor can
// contribute a local predecessor, so LOW/HIGH require at most their factor
// width entries. Never silently truncate a wider production build.
static_assert(LOW_LUT_K<=BKCZ_MAX_LOCAL,"zero-closure LOW local source cache too small");
static_assert(HIGH_LUT_K<=BKCZ_MAX_LOCAL,"zero-closure HIGH local source cache too small");
struct BkczPlan{
    uint32_t local[BKCZ_MAX_LOCAL];
    uint32_t cross_src=0;
    uint8_t local_n=0,cross_depth=0,cross_valid=0,pad=0;
};

__device__ __forceinline__ void bkcz_plan_add_low(BkczPlan&p,MateID x,int fixed_he){uint32_t loc=0,bid=0;if(bkcz_low_source_ref(x,fixed_he,loc,bid)&&p.local_n<BKCZ_MAX_LOCAL)p.local[p.local_n++]=bkf_src_pack(bid,loc);}
__device__ __forceinline__ void bkcz_plan_add_high(BkczPlan&p,MateID x,int fixed_hs){uint32_t loc=0,bid=0;if(bkcz_high_source_ref(x,fixed_hs,loc,bid)&&p.local_n<BKCZ_MAX_LOCAL)p.local[p.local_n++]=bkf_src_pack(bid,loc);}

__device__ __forceinline__ BkczPlan bkcz_build_low_plan(MateID d,int fixed_he,int p){
    BkczPlan z{};if(mpair(d,p)!=NN)return z;MateID x=msetpair(d,p,RL);bkcz_plan_add_low(z,x,fixed_he);int bal=0;for(int q=p-2;q>=0;--q){MateValue v=mget(d,q);if(bal==0&&v==L){x=msetpair(d,p,LL);x=mset(x,q,R);bkcz_plan_add_low(z,x,fixed_he);}if(v==L)++bal;else if(v==R)--bal;if(bal<0)break;}bal=0;for(int q=p+1;q<LOW_LUT_K+1;++q){MateValue v=mget(d,q);if(bal==0&&v==R){x=msetpair(d,p,RR);x=mset(x,q,L);bkcz_plan_add_low(z,x,fixed_he);}if(v==R)++bal;else if(v==L)--bal;if(bal<0)break;}
    int depth=low_cross_preimage_partial(d,LOW_LUT_K+1,p,x);if(depth>0){uint32_t loc=0,bid=0;if(bkcz_low_source_ref(x,fixed_he+2,loc,bid)){z.cross_src=bkf_src_pack(bid,loc);z.cross_depth=uint8_t(depth);z.cross_valid=1;}}return z;
}
__device__ __forceinline__ BkczPlan bkcz_build_high_plan(MateID d,int fixed_hs,int rel){
    BkczPlan z{};if(mpair(d,rel)!=NN)return z;MateID x=msetpair(d,rel,RL);bkcz_plan_add_high(z,x,fixed_hs);int bal=0;for(int q=rel+1;q<HIGH_LUT_K+1;++q){MateValue v=mget(d,q);if(bal==0&&v==R){x=msetpair(d,rel,RR);x=mset(x,q,L);bkcz_plan_add_high(z,x,fixed_hs);}if(v==R)++bal;else if(v==L)--bal;if(bal<0)break;}bal=0;for(int q=rel-2;q>=0;--q){MateValue v=mget(d,q);if(bal==0&&v==L){x=msetpair(d,rel,LL);x=mset(x,q,R);bkcz_plan_add_high(z,x,fixed_hs);}if(v==L)++bal;else if(v==R)--bal;if(bal<0)break;}
    int depth=high_cross_preimage_partial(d,HIGH_LUT_K+1,rel,x);if(depth>0){uint32_t loc=0,bid=0;if(bkcz_high_source_ref(x,fixed_hs+2,loc,bid)){z.cross_src=bkf_src_pack(bid,loc);z.cross_depth=uint8_t(depth);z.cross_valid=1;}}return z;
}

__device__ __forceinline__ BkczPlan bkcz_forward_low_plan(uint32_t dest_loc,const BucketPhysicalBlock&db,int p){uint32_t dc=bkci_low_code(dest_loc,db.hs);MateID d=p==1?(MateID(dc)|(MateID(db.c)<<(2*LOW_LUT_K))):minsert(MateID(dc),p,N);return bkcz_build_low_plan(d,db.he,p);}
__device__ __forceinline__ BkczPlan bkcz_forward_high_plan(uint32_t dest_loc,const BucketPhysicalBlock&db,int p){uint32_t dc=bkci_high_code(dest_loc,db.he);int rel=p-LOW_LUT_K;MateID d=minsert(MateID(dc),rel,N);return bkcz_build_high_plan(d,db.hs,rel);}
__device__ __forceinline__ BkczPlan bkcz_reverse_low_plan(uint32_t dest_loc,const BucketPhysicalBlock&db,int p){uint32_t dc=bkci_low_code(dest_loc,db.hs);MateID d=blocked_exclude_reverse(MateID(dc),LOW_LUT_K+1,p);return bkcz_build_low_plan(d,db.he,p);}
__device__ __forceinline__ BkczPlan bkcz_reverse_high_plan(uint32_t dest_loc,const BucketPhysicalBlock&db,int p,bool edge){uint32_t dc=bkci_high_code(dest_loc,db.he);int rel=p-LOW_LUT_K;MateID d=edge?(MateID(db.c)|(MateID(dc)<<2)):blocked_exclude_reverse(MateID(dc),HIGH_LUT_K+1,rel);return bkcz_build_high_plan(d,db.hs,rel);}

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
    if(p.cross_valid){uint32_t x=p.cross_src,sl=bkf_src_locator(x);BucketPhysicalBlock sb=bkf_low_main(bkf_loc_owner(sl),bkf_src_block(x));uint32_t dc=D_BKF_HIGH_CODES[D_BKF_HIGH_CODE_OFF[size_t(D_BKF_FIXED_OWNER)*D_BKF_CODE_PITCH+db.he]+hr];
#if GPU_DIRECT_PM_ACCUM
        sum+=bkf_sum_high_preimages_u64(dc,p.cross_depth,sb.he,bkf_src_block(x),sl);
#else
        sum=gpu_direct_add(sum,bkf_sum_high_preimages(dc,p.cross_depth,sb.he,bkf_src_block(x),sl));
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
    if(p.cross_valid){uint32_t x=p.cross_src,sl=bkf_src_locator(x);BucketPhysicalBlock sb=bkf_high_main(bkf_loc_owner(sl),bkf_src_block(x));uint32_t dc=D_BKF_LOW_CODES[D_BKF_LOW_CODE_OFF[size_t(D_BKF_FIXED_OWNER)*D_BKF_CODE_PITCH+db.hs]+lr];
#if GPU_DIRECT_PM_ACCUM
        sum+=bkf_sum_low_preimages_u64(dc,p.cross_depth,sb.hs,bkf_src_block(x),sl);
#else
        sum=gpu_direct_add(sum,bkf_sum_low_preimages(dc,p.cross_depth,sb.hs,bkf_src_block(x),sl));
#endif
    }
#if GPU_DIRECT_PM_ACCUM
    return gpu_direct_pm_reduce_u64(sum);
#else
    return sum;
#endif
}
