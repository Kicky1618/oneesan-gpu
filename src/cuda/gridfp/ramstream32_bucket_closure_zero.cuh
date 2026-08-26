#pragma once

#include "ramstream32_bucket_closure_inverse.cuh"

// Metadata-free closure contribution. Both ordinary and CROSS active-half
// sources are reconstructed from the destination partial frontier. CROSS then
// uses the existing inactive-half ternary inverse walker. No destination record,
// source CSR, CROSS op or closure attachment is required at runtime.

__device__ __forceinline__ bool bkcz_low_source_ref(MateID partial,int fixed_he,uint32_t&loc,uint32_t&bid){
    constexpr uint64_t MASK=(uint64_t(1)<<(2*LOW_LUT_K))-1ull;uint32_t c=uint32_t(mget(partial,LOW_LUT_K)),lc=uint32_t(partial&MASK);int hs=fixed_he+bkci_delta(c);if(fixed_he<0||fixed_he>HIGH_LUT_K+1||hs<0||hs>LOW_LUT_K+1)return false;uint32_t z=D_BKF_LOW_DIRECT[gdx_ternary_key<LOW_LUT_K>(lc)];if(z==BKF_DIRECT_INVALID||int(bkf_direct_height(z))!=hs)return false;loc=bkf_direct_locator(z);bid=uint32_t(3*fixed_he+int(c));return bid<D_BKF_MAIN_NBLOCKS&&bkf_low_main(bkf_loc_owner(loc),bid).valid;
}
__device__ __forceinline__ bool bkcz_high_source_ref(MateID partial,int fixed_hs,uint32_t&loc,uint32_t&bid){
    constexpr uint64_t MASK=(uint64_t(1)<<(2*HIGH_LUT_K))-1ull;uint32_t c=uint32_t(mget(partial,0)),hc=uint32_t((partial>>2)&MASK);int he=fixed_hs-bkci_delta(c);if(fixed_hs<0||fixed_hs>LOW_LUT_K+1||he<0||he>HIGH_LUT_K+1)return false;uint32_t z=D_BKF_HIGH_DIRECT[gdx_ternary_key<HIGH_LUT_K>(hc)];if(z==BKF_DIRECT_INVALID||int(bkf_direct_height(z))!=he)return false;loc=bkf_direct_locator(z);bid=uint32_t(3*he+int(c));return bid<D_BKF_MAIN_NBLOCKS&&bkf_high_main(bkf_loc_owner(loc),bid).valid;
}

#if GPU_DIRECT_PM_ACCUM
__device__ __forceinline__ uint64_t bkcz_low_ordinary_raw(MateID d,int fixed_he,uint32_t hr,int p){
    if(mpair(d,p)!=NN)return 0;uint64_t s=0;MateID x=msetpair(d,p,RL);s+=bkci_low_source_value_u64(x,fixed_he,hr);int bal=0;for(int q=p-2;q>=0;--q){MateValue v=mget(d,q);if(bal==0&&v==L){x=msetpair(d,p,LL);x=mset(x,q,R);s+=bkci_low_source_value_u64(x,fixed_he,hr);}if(v==L)++bal;else if(v==R)--bal;if(bal<0)break;}bal=0;for(int q=p+1;q<LOW_LUT_K+1;++q){MateValue v=mget(d,q);if(bal==0&&v==R){x=msetpair(d,p,RR);x=mset(x,q,L);s+=bkci_low_source_value_u64(x,fixed_he,hr);}if(v==R)++bal;else if(v==L)--bal;if(bal<0)break;}return s;
}
__device__ __forceinline__ uint64_t bkcz_high_ordinary_raw(MateID d,int fixed_hs,uint32_t lr,int rel){
    if(mpair(d,rel)!=NN)return 0;uint64_t s=0;MateID x=msetpair(d,rel,RL);s+=bkci_high_source_value_u64(x,fixed_hs,lr);int bal=0;for(int q=rel+1;q<HIGH_LUT_K+1;++q){MateValue v=mget(d,q);if(bal==0&&v==R){x=msetpair(d,rel,RR);x=mset(x,q,L);s+=bkci_high_source_value_u64(x,fixed_hs,lr);}if(v==R)++bal;else if(v==L)--bal;if(bal<0)break;}bal=0;for(int q=rel-2;q>=0;--q){MateValue v=mget(d,q);if(bal==0&&v==L){x=msetpair(d,rel,LL);x=mset(x,q,R);s+=bkci_high_source_value_u64(x,fixed_hs,lr);}if(v==L)++bal;else if(v==R)--bal;if(bal<0)break;}return s;
}
#else
__device__ __forceinline__ Count bkcz_low_ordinary(MateID d,int fixed_he,uint32_t hr,int p){
    if(mpair(d,p)!=NN)return 0;Count s=0;MateID x=msetpair(d,p,RL);s=gpu_direct_add(s,bkci_low_source_value(x,fixed_he,hr));int bal=0;for(int q=p-2;q>=0;--q){MateValue v=mget(d,q);if(bal==0&&v==L){x=msetpair(d,p,LL);x=mset(x,q,R);s=gpu_direct_add(s,bkci_low_source_value(x,fixed_he,hr));}if(v==L)++bal;else if(v==R)--bal;if(bal<0)break;}bal=0;for(int q=p+1;q<LOW_LUT_K+1;++q){MateValue v=mget(d,q);if(bal==0&&v==R){x=msetpair(d,p,RR);x=mset(x,q,L);s=gpu_direct_add(s,bkci_low_source_value(x,fixed_he,hr));}if(v==R)++bal;else if(v==L)--bal;if(bal<0)break;}return s;
}
__device__ __forceinline__ Count bkcz_high_ordinary(MateID d,int fixed_hs,uint32_t lr,int rel){
    if(mpair(d,rel)!=NN)return 0;Count s=0;MateID x=msetpair(d,rel,RL);s=gpu_direct_add(s,bkci_high_source_value(x,fixed_hs,lr));int bal=0;for(int q=rel+1;q<HIGH_LUT_K+1;++q){MateValue v=mget(d,q);if(bal==0&&v==R){x=msetpair(d,rel,RR);x=mset(x,q,L);s=gpu_direct_add(s,bkci_high_source_value(x,fixed_hs,lr));}if(v==R)++bal;else if(v==L)--bal;if(bal<0)break;}bal=0;for(int q=rel-2;q>=0;--q){MateValue v=mget(d,q);if(bal==0&&v==L){x=msetpair(d,rel,LL);x=mset(x,q,R);s=gpu_direct_add(s,bkci_high_source_value(x,fixed_hs,lr));}if(v==L)++bal;else if(v==R)--bal;if(bal<0)break;}return s;
}
#endif

__device__ __forceinline__ Count bkcz_low_extra_partial(MateID d,const BucketPhysicalBlock&db,uint32_t hr,int p){
#if GPU_DIRECT_PM_ACCUM
    uint64_t sum=bkcz_low_ordinary_raw(d,db.he,hr,p);
#else
    Count sum=bkcz_low_ordinary(d,db.he,hr,p);
#endif
    MateID src=0;int depth=low_cross_preimage_partial(d,LOW_LUT_K+1,p,src);if(depth>0){uint32_t sl=0,sbid=0;int she=int(db.he)+2;if(bkcz_low_source_ref(src,she,sl,sbid)){uint32_t dc=D_BKF_HIGH_CODES[D_BKF_HIGH_CODE_OFF[size_t(D_BKF_FIXED_OWNER)*D_BKF_CODE_PITCH+db.he]+hr];
#if GPU_DIRECT_PM_ACCUM
        sum+=bkf_sum_high_preimages_u64(dc,uint32_t(depth),she,sbid,sl);
#else
        sum=gpu_direct_add(sum,bkf_sum_high_preimages(dc,uint32_t(depth),she,sbid,sl));
#endif
    }}
#if GPU_DIRECT_PM_ACCUM
    return gpu_direct_pm_reduce_u64(sum);
#else
    return sum;
#endif
}

__device__ __forceinline__ Count bkcz_high_extra_partial(MateID d,const BucketPhysicalBlock&db,uint32_t lr,int rel){
#if GPU_DIRECT_PM_ACCUM
    uint64_t sum=bkcz_high_ordinary_raw(d,db.hs,lr,rel);
#else
    Count sum=bkcz_high_ordinary(d,db.hs,lr,rel);
#endif
    MateID src=0;int depth=high_cross_preimage_partial(d,HIGH_LUT_K+1,rel,src);if(depth>0){uint32_t sl=0,sbid=0;int shs=int(db.hs)+2;if(bkcz_high_source_ref(src,shs,sl,sbid)){uint32_t dc=D_BKF_LOW_CODES[D_BKF_LOW_CODE_OFF[size_t(D_BKF_FIXED_OWNER)*D_BKF_CODE_PITCH+db.hs]+lr];
#if GPU_DIRECT_PM_ACCUM
        sum+=bkf_sum_low_preimages_u64(dc,uint32_t(depth),shs,sbid,sl);
#else
        sum=gpu_direct_add(sum,bkf_sum_low_preimages(dc,uint32_t(depth),shs,sbid,sl));
#endif
    }}
#if GPU_DIRECT_PM_ACCUM
    return gpu_direct_pm_reduce_u64(sum);
#else
    return sum;
#endif
}

__device__ __forceinline__ Count bkcz_forward_low_extra(uint32_t dest_loc,const BucketPhysicalBlock&db,uint32_t hr,int p){uint32_t dc=bkci_low_code(dest_loc,db.hs);MateID d=p==1?(MateID(dc)|(MateID(db.c)<<(2*LOW_LUT_K))):minsert(MateID(dc),p,N);return bkcz_low_extra_partial(d,db,hr,p);}
__device__ __forceinline__ Count bkcz_forward_high_extra(uint32_t dest_loc,const BucketPhysicalBlock&db,uint32_t lr,int p){uint32_t dc=bkci_high_code(dest_loc,db.he);int rel=p-LOW_LUT_K;MateID d=minsert(MateID(dc),rel,N);return bkcz_high_extra_partial(d,db,lr,rel);}
__device__ __forceinline__ Count bkcz_reverse_low_extra(uint32_t dest_loc,const BucketPhysicalBlock&db,uint32_t hr,int p){uint32_t dc=bkci_low_code(dest_loc,db.hs);MateID d=blocked_exclude_reverse(MateID(dc),LOW_LUT_K+1,p);return bkcz_low_extra_partial(d,db,hr,p);}
__device__ __forceinline__ Count bkcz_reverse_high_extra(uint32_t dest_loc,const BucketPhysicalBlock&db,uint32_t lr,int p,bool edge){uint32_t dc=bkci_high_code(dest_loc,db.he);int rel=p-LOW_LUT_K;MateID d=edge?(MateID(db.c)|(MateID(dc)<<2)):blocked_exclude_reverse(MateID(dc),HIGH_LUT_K+1,rel);return bkcz_high_extra_partial(d,db,lr,rel);}
