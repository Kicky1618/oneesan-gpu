#pragma once

#include "../../common/gridfp_closure_pattern10.hpp"
#include "ramstream32_bucket_closure_zero_plan.cuh"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>

static constexpr uint64_t BKCP10_BASE_MASK=(1ull<<54)-1ull;
static_assert(LOW_LUT_K<=14&&HIGH_LUT_K<=14,"pattern10 backend requires factor widths <=14");

#if defined(__CUDACC__)
#define BKCP10_HD __host__ __device__ __forceinline__
#else
#define BKCP10_HD inline
#endif
BKCP10_HD uint16_t bkcp10_id(BucketOrbitOp op){return uint16_t((op>>54)&0x3ffu);}
#undef BKCP10_HD

static uint32_t bkcp10_low_code_host(const BucketFusedHost&f,uint32_t loc,int h){
    uint32_t g=bkf_loc_owner(loc),r=bkf_loc_rank(loc);size_t ix=size_t(g)*f.code_pitch+uint32_t(h);
    if(ix>=f.low_code_off.size())std::exit(560);uint32_t q=f.low_code_off[ix]+r;if(q>=f.low_codes.size())std::exit(561);return f.low_codes[q];
}
static uint32_t bkcp10_high_code_host(const BucketFusedHost&f,uint32_t loc,int h){
    uint32_t g=bkf_loc_owner(loc),r=bkf_loc_rank(loc);size_t ix=size_t(g)*f.code_pitch+uint32_t(h);
    if(ix>=f.high_code_off.size())std::exit(562);uint32_t q=f.high_code_off[ix]+r;if(q>=f.high_codes.size())std::exit(563);return f.high_codes[q];
}
static BucketOrbitOp bkcp10_set(BucketOrbitOp op,uint16_t id){
    if(id>0x3ffu)std::exit(564);return (op&BKCP10_BASE_MASK)|(uint64_t(id)<<54);
}

static void build_bucket_forward_pattern10(
    const StorageLayout&layout,BucketOrbitStreamsHost&bo,const BucketFusedHost&bf
){
    using namespace oneesan::gridfp;size_t lp=size_t(bo.low_nblocks)+1,hp=size_t(bo.high_nblocks)+1;uint64_t encoded=0,none=0;uint16_t maxid=0;
    auto enc_low=[&](BucketOrbitOp&op,const StorageBlock&db,int p,bool active){uint16_t id=CLOSURE_PATTERN10_NONE;if(active){uint32_t loc=p==1?bkf_orbit_src(op):bkf_orbit_drop(op);uint32_t dc=bkcp10_low_code_host(bf,loc,db.hs);MateID d=p==1?(MateID(dc)|(MateID(db.c)<<(2*LOW_LUT_K))):minsert(MateID(dc),p,N);id=closure_pattern10_encode(d,LOW_LUT_K+1,p);}op=bkcp10_set(op,id);if(id==CLOSURE_PATTERN10_NONE)++none;else{++encoded;maxid=std::max(maxid,id);}};
    for(int p=LOW_LUT_K;p>=1;--p){uint32_t pi=uint32_t(LOW_LUT_K-p);for(uint32_t bid=0;bid<bo.low_nblocks;++bid){const StorageBlock&xb=layout.main_blocks[bid];if(!xb.valid)continue;const StorageBlock&db=p==1?xb:layout.block_blocks[xb.he];auto scan=[&](std::vector<BucketOrbitOp>&v,const std::vector<uint32_t>&off,bool active){uint32_t a=off[size_t(pi)*lp+bid],b=off[size_t(pi)*lp+bid+1];for(uint32_t q=a;q<b;++q)enc_low(v[q],db,p,active);};scan(bo.low_nn,bo.low_nn_off,true);scan(bo.low_nr,bo.low_nr_off,p!=1);scan(bo.low_nl,bo.low_nl_off,p!=1);}}
    auto enc_high=[&](BucketOrbitOp&op,const StorageBlock&db,int p){uint32_t loc=bkf_orbit_drop(op),dc=bkcp10_high_code_host(bf,loc,db.he);int rel=p-LOW_LUT_K;MateID d=minsert(MateID(dc),rel,N);uint16_t id=closure_pattern10_encode(d,HIGH_LUT_K+1,rel);op=bkcp10_set(op,id);if(id==CLOSURE_PATTERN10_NONE)++none;else{++encoded;maxid=std::max(maxid,id);}};
    for(int p=TARGET_W-1;p>=LOW_LUT_K+1;--p){uint32_t pi=uint32_t((TARGET_W-1)-p);for(uint32_t bid=0;bid<bo.high_nblocks;++bid){const StorageBlock&xb=layout.main_blocks[bid];if(!xb.valid)continue;const StorageBlock&db=layout.block_blocks[xb.hs];auto scan=[&](std::vector<BucketOrbitOp>&v,const std::vector<uint32_t>&off){uint32_t a=off[size_t(pi)*hp+bid],b=off[size_t(pi)*hp+bid+1];for(uint32_t q=a;q<b;++q)enc_high(v[q],db,p);};scan(bo.high_nn,bo.high_nn_off);scan(bo.high_nrnl,bo.high_nrnl_off);}}
    std::cerr<<"bucket_forward_pattern10 encoded="<<encoded<<" none="<<none<<" max_id="<<maxid<<" extra_bytes=0\n";
}

static void build_reverse_split54_pattern10(
    const StorageLayout&layout,const BucketFusedHost&bf,ReverseSplit54Host&sp
){
    using namespace oneesan::gridfp;size_t pitch=size_t(sp.nblocks)+1;uint64_t encoded=0,none=0;uint16_t maxid=0;
    auto setid=[&](BucketOrbitOp&op,uint16_t id){op=bkcp10_set(op,id);if(id==CLOSURE_PATTERN10_NONE)++none;else{++encoded;maxid=std::max(maxid,id);}};
    for(int p=1;p<=LOW_LUT_K;++p){uint32_t pi=uint32_t(p-1);for(uint32_t bid=0;bid<sp.nblocks;++bid){const StorageBlock&xb=layout.main_blocks[bid];if(!xb.valid)continue;const StorageBlock&db=layout.block_blocks[xb.he];auto scan=[&](std::vector<BucketOrbitOp>&v,const std::vector<uint32_t>&off){uint32_t a=off[size_t(pi)*pitch+bid],b=off[size_t(pi)*pitch+bid+1];for(uint32_t q=a;q<b;++q){uint32_t loc=bkf_orbit_drop(v[q]),dc=bkcp10_low_code_host(bf,loc,db.hs);MateID d=blocked_exclude_reverse(MateID(dc),LOW_LUT_K+1,p);setid(v[q],closure_pattern10_encode(d,LOW_LUT_K+1,p));}};scan(sp.low.nn,sp.low.nn_off);scan(sp.low.nr,sp.low.nr_off);scan(sp.low.nl,sp.low.nl_off);}}
    for(int p=LOW_LUT_K+1;p<TARGET_W;++p){uint32_t pi=uint32_t(p-(LOW_LUT_K+1));bool edge=p==TARGET_W-1;int rel=p-LOW_LUT_K;for(uint32_t bid=0;bid<sp.nblocks;++bid){const StorageBlock&xb=layout.main_blocks[bid];if(!xb.valid)continue;auto scan=[&](std::vector<BucketOrbitOp>&v,const std::vector<uint32_t>&off,bool nn){uint32_t a=off[size_t(pi)*pitch+bid],b=off[size_t(pi)*pitch+bid+1];for(uint32_t q=a;q<b;++q){uint16_t id=CLOSURE_PATTERN10_NONE;if(!edge||nn){uint32_t loc=edge?bkf_orbit_src(v[q]):bkf_orbit_drop(v[q]);const StorageBlock&db=edge?xb:layout.block_blocks[xb.hs];uint32_t dc=bkcp10_high_code_host(bf,loc,db.he);MateID d=edge?(MateID(db.c)|(MateID(dc)<<2)):blocked_exclude_reverse(MateID(dc),HIGH_LUT_K+1,rel);id=closure_pattern10_encode(d,HIGH_LUT_K+1,rel);}setid(v[q],id);}};scan(sp.high.nn,sp.high.nn_off,true);scan(sp.high.nr,sp.high.nr_off,false);scan(sp.high.nl,sp.high.nl_off,false);}}
    std::cerr<<"reverse_split54_pattern10 encoded="<<encoded<<" none="<<none<<" max_id="<<maxid<<" extra_bytes=0\n";
}

__device__ __forceinline__ BkczPlan bkcp10_build_low_plan(MateID d,int fixed_he,int p,uint16_t id){
    using namespace oneesan::gridfp;BkczPlan z{};if(id==CLOSURE_PATTERN10_NONE||mpair(d,p)!=NN)return z;MateID x=msetpair(d,p,RL);bkcz_plan_add_low(z,x,fixed_he);uint16_t lm=0,rm=0;closure_pattern10_decode(id,LOW_LUT_K+1,p,lm,rm);for(int i=0;i<p-1;++i)if((lm>>i)&1u){int q=p-2-i;x=msetpair(d,p,LL);x=mset(x,q,R);bkcz_plan_add_low(z,x,fixed_he);}for(int i=0;i<LOW_LUT_K-p;++i)if((rm>>i)&1u){int q=p+1+i;x=msetpair(d,p,RR);x=mset(x,q,L);bkcz_plan_add_low(z,x,fixed_he);}int depth=low_cross_preimage_partial(d,LOW_LUT_K+1,p,x);if(depth>0){uint32_t loc=0,bid=0;if(bkcz_low_source_ref(x,fixed_he+2,loc,bid)){z.cross_src=bkf_src_pack(bid,loc);z.cross_depth=uint8_t(depth);z.cross_valid=1;}}return z;
}
__device__ __forceinline__ BkczPlan bkcp10_build_high_plan(MateID d,int fixed_hs,int rel,uint16_t id){
    using namespace oneesan::gridfp;BkczPlan z{};if(id==CLOSURE_PATTERN10_NONE||mpair(d,rel)!=NN)return z;MateID x=msetpair(d,rel,RL);bkcz_plan_add_high(z,x,fixed_hs);uint16_t lm=0,rm=0;closure_pattern10_decode(id,HIGH_LUT_K+1,rel,lm,rm);for(int i=0;i<rel-1;++i)if((lm>>i)&1u){int q=rel-2-i;x=msetpair(d,rel,LL);x=mset(x,q,R);bkcz_plan_add_high(z,x,fixed_hs);}for(int i=0;i<HIGH_LUT_K-rel;++i)if((rm>>i)&1u){int q=rel+1+i;x=msetpair(d,rel,RR);x=mset(x,q,L);bkcz_plan_add_high(z,x,fixed_hs);}int depth=high_cross_preimage_partial(d,HIGH_LUT_K+1,rel,x);if(depth>0){uint32_t loc=0,bid=0;if(bkcz_high_source_ref(x,fixed_hs+2,loc,bid)){z.cross_src=bkf_src_pack(bid,loc);z.cross_depth=uint8_t(depth);z.cross_valid=1;}}return z;
}
