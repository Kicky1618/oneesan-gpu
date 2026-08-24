#pragma once

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

// Bucket-local version of the zero-scratch fused direct executor.
//
// The physical state on one GPU is eight raw bucket slots.  During the LOW
// window the GPU id is the fixed HIGH owner and slot selects the active LOW
// owner: pair[fixed_h][slot_l].  During the HIGH window the GPU id is the fixed
// LOW owner and slot selects the active HIGH owner: pair[slot_h][fixed_l].
// Active-half source/partner/drop and closure ranks are 18-bit bucket locators
// (3 owner bits + 15 owner-local rank bits).  CROSS repair preserves inactive
// occupancy, therefore every generated inactive-half preimage stays on the
// current GPU; no peer access occurs inside a window.

static constexpr uint32_t BKF_LOC_MASK=(1u<<BUCKET_LOCATOR_BITS)-1u;
static constexpr int BKF_SRC_BLOCK_SHIFT=BUCKET_LOCATOR_BITS;
static constexpr uint32_t BKF_SRC_BLOCK_MASK=0x3fu;
static constexpr int BKF_CROSS_DEPTH_SHIFT=BUCKET_LOCATOR_BITS+6;
static constexpr uint32_t BKF_CROSS_DEPTH_MASK=0x0fu;
static constexpr uint32_t BKF_DIRECT_INVALID=0xffffffffu;
static constexpr int BKF_DIRECT_HEIGHT_SHIFT=BUCKET_LOCATOR_BITS;
static constexpr uint32_t BKF_DIRECT_HEIGHT_MASK=0x1fu;

static inline uint32_t bkf_src_pack(uint32_t bid,uint32_t loc){
    if(bid>BKF_SRC_BLOCK_MASK||(loc&~BKF_LOC_MASK)){
        std::cerr<<"bucket fused source overflow block="<<bid<<" loc="<<loc<<'\n';
        std::exit(200);
    }
    return loc|(bid<<BKF_SRC_BLOCK_SHIFT);
}
static inline uint32_t bkf_cross_pack(uint32_t bid,uint32_t loc,uint32_t depth){
    if(!depth||depth>BKF_CROSS_DEPTH_MASK){
        std::cerr<<"bucket fused depth overflow depth="<<depth<<'\n';
        std::exit(201);
    }
    return bkf_src_pack(bid,loc)|(depth<<BKF_CROSS_DEPTH_SHIFT);
}
static inline uint32_t bkf_src_locator(uint32_t x){return x&BKF_LOC_MASK;}
static inline uint32_t bkf_src_block(uint32_t x){return (x>>BKF_SRC_BLOCK_SHIFT)&BKF_SRC_BLOCK_MASK;}
static inline uint32_t bkf_cross_depth(uint32_t x){return (x>>BKF_CROSS_DEPTH_SHIFT)&BKF_CROSS_DEPTH_MASK;}

struct BucketFusedDst{
    uint32_t dst_locator=0;
    uint32_t local_begin=0;
    uint32_t cross_begin=0;
    uint32_t counts=0;
};
static_assert(sizeof(BucketFusedDst)==16);

struct BucketFusedHost{
    std::vector<BucketFusedDst> low_dst;
    std::vector<uint32_t> low_off;
    uint32_t low_pitch=GPU_DIRECT_MAX_MAIN_BLOCKS+1;
    std::vector<uint32_t> low_local_src;
    std::vector<uint32_t> low_cross_op;

    std::vector<BucketFusedDst> high_dst;
    std::vector<uint32_t> high_off;
    uint32_t high_pitch=GPU_DIRECT_MAX_BLOCK_BLOCKS+1;
    std::vector<uint32_t> high_local_src;
    std::vector<uint32_t> high_cross_op;

    // owner-major, then height-major, then owner-local rank.
    std::vector<uint32_t> high_codes,low_codes;
    std::vector<uint32_t> high_code_off,low_code_off;
    uint32_t code_pitch=MAXW+2;

    // ternary code -> (height << 18) | locator.
    std::vector<uint32_t> high_direct,low_direct;

    size_t bytes()const{
        return (low_dst.size()+high_dst.size())*sizeof(BucketFusedDst)
            +(low_off.size()+high_off.size()+low_local_src.size()+low_cross_op.size()
              +high_local_src.size()+high_cross_op.size()+high_codes.size()+low_codes.size()
              +high_code_off.size()+low_code_off.size()+high_direct.size()+low_direct.size())
                *sizeof(uint32_t);
    }
};

static BucketFusedHost build_bucket_fused(
    const StorageFactorHost&storage,const StorageLayout&layout,
    const BucketOwnerHost&owner,const GpuDirectGatherHost&ordinary,
    const GpuDirectCrossGatherHost&cross,const GpuDirectFusedHost&fused
){
    BucketFusedHost out;
    out.low_pitch=fused.low_pitch;out.high_pitch=fused.high_pitch;
    out.low_off=fused.low_off;out.high_off=fused.high_off;

    out.low_local_src.reserve(ordinary.low_src.size());
    for(uint32_t x:ordinary.low_src){
        uint32_t bid=gpu_direct_gather_src_block(x),r=gpu_direct_gather_src_rank(x);
        const StorageBlock&b=layout.main_blocks[bid];
        out.low_local_src.push_back(bkf_src_pack(bid,bucket_low_locator(storage,owner,b.hs,r)));
    }
    out.high_local_src.reserve(ordinary.high_src.size());
    for(uint32_t x:ordinary.high_src){
        uint32_t bid=gpu_direct_gather_src_block(x),r=gpu_direct_gather_src_rank(x);
        const StorageBlock&b=layout.main_blocks[bid];
        out.high_local_src.push_back(bkf_src_pack(bid,bucket_high_locator(storage,owner,b.he,r)));
    }
    out.low_cross_op.reserve(cross.low_op.size());
    for(uint32_t x:cross.low_op){
        uint32_t bid=gdx_op_block(x),r=gdx_op_rank(x),d=gdx_op_depth(x);
        const StorageBlock&b=layout.main_blocks[bid];
        out.low_cross_op.push_back(bkf_cross_pack(bid,bucket_low_locator(storage,owner,b.hs,r),d));
    }
    out.high_cross_op.reserve(cross.high_op.size());
    for(uint32_t x:cross.high_op){
        uint32_t bid=gdx_op_block(x),r=gdx_op_rank(x),d=gdx_op_depth(x);
        const StorageBlock&b=layout.main_blocks[bid];
        out.high_cross_op.push_back(bkf_cross_pack(bid,bucket_high_locator(storage,owner,b.he,r),d));
    }

    out.low_dst.reserve(fused.low_dst.size());
    for(int p=LOW_LUT_K;p>=1;--p){
        uint32_t pi=uint32_t(LOW_LUT_K-p);
        bool target_main=p==1;
        uint32_t nt=target_main?uint32_t(layout.main_blocks.size()):uint32_t(layout.block_blocks.size());
        for(uint32_t bid=0;bid<nt;++bid){
            uint32_t a=fused.low_off[size_t(pi)*fused.low_pitch+bid];
            uint32_t b=fused.low_off[size_t(pi)*fused.low_pitch+bid+1];
            const StorageBlock&db=target_main?layout.main_blocks[bid]:layout.block_blocks[bid];
            for(uint32_t q=a;q<b;++q){
                const GpuDirectFusedDst&r=fused.low_dst[q];
                out.low_dst.push_back({bucket_low_locator(storage,owner,db.hs,r.dst_rank),
                    r.local_begin,r.cross_begin,r.counts});
            }
        }
    }
    out.high_dst.reserve(fused.high_dst.size());
    for(uint32_t pi=0;pi<uint32_t(HIGH_LUT_K);++pi){
        for(uint32_t bid=0;bid<uint32_t(layout.block_blocks.size());++bid){
            uint32_t a=fused.high_off[size_t(pi)*fused.high_pitch+bid];
            uint32_t b=fused.high_off[size_t(pi)*fused.high_pitch+bid+1];
            const StorageBlock&db=layout.block_blocks[bid];
            for(uint32_t q=a;q<b;++q){
                const GpuDirectFusedDst&r=fused.high_dst[q];
                out.high_dst.push_back({bucket_high_locator(storage,owner,db.he,r.dst_rank),
                    r.local_begin,r.cross_begin,r.counts});
            }
        }
    }
    if(out.low_dst.size()!=fused.low_dst.size()||out.high_dst.size()!=fused.high_dst.size()){
        std::cerr<<"bucket fused destination size mismatch\n";std::exit(202);
    }

    const uint32_t P=uint32_t(MAXW+2);
    out.high_code_off.assign(BUCKET_NGPU*P,0);
    out.low_code_off.assign(BUCKET_NGPU*P,0);
    uint32_t z=0;
    for(uint32_t g=0;g<BUCKET_NGPU;++g)for(uint32_t h=0;h<P;++h){
        out.high_code_off[size_t(g)*P+h]=z;
        if(h<uint32_t(MAXW+1))z+=owner.high_count[g][h];
    }
    out.high_codes.assign(z,0xffffffffu);
    z=0;
    for(uint32_t g=0;g<BUCKET_NGPU;++g)for(uint32_t h=0;h<P;++h){
        out.low_code_off[size_t(g)*P+h]=z;
        if(h<uint32_t(MAXW+1))z+=owner.low_count[g][h];
    }
    out.low_codes.assign(z,0xffffffffu);

    out.high_direct.assign(gpu_direct_pow3(HIGH_LUT_K),BKF_DIRECT_INVALID);
    for(int h=0;h<=HIGH_LUT_K+1;++h){
        uint32_t a=storage.high_all_off[h],b=storage.high_all_off[h+1];
        for(uint32_t gi=a;gi<b;++gi){
            uint32_t loc=owner.high_all_locator[gi],g=bucket_locator_owner(loc),r=bucket_locator_rank(loc);
            out.high_codes[out.high_code_off[size_t(g)*P+h]+r]=storage.high_all_codes[gi];
            uint32_t key=gpu_direct_ternary_key_host(storage.high_all_codes[gi],HIGH_LUT_K);
            if(out.high_direct[key]!=BKF_DIRECT_INVALID){std::cerr<<"bucket fused duplicate HIGH code\n";std::exit(203);}
            out.high_direct[key]=(uint32_t(h)<<BKF_DIRECT_HEIGHT_SHIFT)|loc;
        }
    }
    out.low_direct.assign(gpu_direct_pow3(LOW_LUT_K),BKF_DIRECT_INVALID);
    for(int h=0;h<=LOW_LUT_K+1;++h){
        uint32_t a=storage.low_all_off[h],b=storage.low_all_off[h+1];
        for(uint32_t gi=a;gi<b;++gi){
            uint32_t loc=owner.low_all_locator[gi],g=bucket_locator_owner(loc),r=bucket_locator_rank(loc);
            out.low_codes[out.low_code_off[size_t(g)*P+h]+r]=storage.low_all_codes[gi];
            uint32_t key=gpu_direct_ternary_key_host(storage.low_all_codes[gi],LOW_LUT_K);
            if(out.low_direct[key]!=BKF_DIRECT_INVALID){std::cerr<<"bucket fused duplicate LOW code\n";std::exit(204);}
            out.low_direct[key]=(uint32_t(h)<<BKF_DIRECT_HEIGHT_SHIFT)|loc;
        }
    }
    for(uint32_t v:out.high_codes)if(v==0xffffffffu){std::cerr<<"bucket fused HIGH code hole\n";std::exit(205);}
    for(uint32_t v:out.low_codes)if(v==0xffffffffu){std::cerr<<"bucket fused LOW code hole\n";std::exit(206);}

    std::cerr<<"bucket_fused low_dst="<<out.low_dst.size()
             <<" high_dst="<<out.high_dst.size()
             <<" low_src="<<out.low_local_src.size()
             <<" high_src="<<out.high_local_src.size()
             <<" low_cross="<<out.low_cross_op.size()
             <<" high_cross="<<out.high_cross_op.size()
             <<" mib="<<double(out.bytes())/double(1<<20)<<'\n';
    return out;
}

__constant__ Count* D_BKF_SLOT[BUCKET_NGPU];
__constant__ uint32_t D_BKF_FIXED_OWNER;
__constant__ BucketPhysicalBlock* D_BKF_LOW_MAIN;
__constant__ BucketPhysicalBlock* D_BKF_LOW_BLOCK;
__constant__ BucketPhysicalBlock* D_BKF_HIGH_MAIN;
__constant__ BucketPhysicalBlock* D_BKF_HIGH_BLOCK;
__constant__ uint32_t D_BKF_MAIN_NBLOCKS;
__constant__ uint32_t D_BKF_BLOCK_NBLOCKS;

__constant__ BucketOrbitOp* D_BKF_LOW_NN;
__constant__ BucketOrbitOp* D_BKF_LOW_NR;
__constant__ BucketOrbitOp* D_BKF_LOW_NL;
__constant__ uint32_t* D_BKF_LOW_NN_OFF;
__constant__ uint32_t* D_BKF_LOW_NR_OFF;
__constant__ uint32_t* D_BKF_LOW_NL_OFF;
__constant__ uint32_t D_BKF_LOW_PITCH;
__constant__ BucketOrbitOp* D_BKF_HIGH_NN;
__constant__ BucketOrbitOp* D_BKF_HIGH_NRNL;
__constant__ uint32_t* D_BKF_HIGH_NN_OFF;
__constant__ uint32_t* D_BKF_HIGH_NRNL_OFF;
__constant__ uint32_t D_BKF_HIGH_PITCH;

__constant__ BucketFusedDst* D_BKF_LOW_DST;
__constant__ uint32_t* D_BKF_LOW_OFF;
__constant__ uint32_t D_BKF_LOW_FUSED_PITCH;
__constant__ uint32_t* D_BKF_LOW_LOCAL_SRC;
__constant__ uint32_t* D_BKF_LOW_CROSS_OP;
__constant__ BucketFusedDst* D_BKF_HIGH_DST;
__constant__ uint32_t* D_BKF_HIGH_OFF;
__constant__ uint32_t D_BKF_HIGH_FUSED_PITCH;
__constant__ uint32_t* D_BKF_HIGH_LOCAL_SRC;
__constant__ uint32_t* D_BKF_HIGH_CROSS_OP;
__constant__ uint32_t* D_BKF_HIGH_CODES;
__constant__ uint32_t* D_BKF_LOW_CODES;
__constant__ uint32_t* D_BKF_HIGH_CODE_OFF;
__constant__ uint32_t* D_BKF_LOW_CODE_OFF;
__constant__ uint32_t D_BKF_CODE_PITCH;
__constant__ uint32_t* D_BKF_HIGH_DIRECT;
__constant__ uint32_t* D_BKF_LOW_DIRECT;

__device__ __forceinline__ BucketPhysicalBlock bkf_low_main(uint32_t slot,uint32_t bid){
    return D_BKF_LOW_MAIN[size_t(slot)*D_BKF_MAIN_NBLOCKS+bid];
}
__device__ __forceinline__ BucketPhysicalBlock bkf_low_block(uint32_t slot,uint32_t bid){
    return D_BKF_LOW_BLOCK[size_t(slot)*D_BKF_BLOCK_NBLOCKS+bid];
}
__device__ __forceinline__ BucketPhysicalBlock bkf_high_main(uint32_t slot,uint32_t bid){
    return D_BKF_HIGH_MAIN[size_t(slot)*D_BKF_MAIN_NBLOCKS+bid];
}
__device__ __forceinline__ BucketPhysicalBlock bkf_high_block(uint32_t slot,uint32_t bid){
    return D_BKF_HIGH_BLOCK[size_t(slot)*D_BKF_BLOCK_NBLOCKS+bid];
}
__device__ __forceinline__ Count* bkf_ptr(uint32_t slot,Code off){return D_BKF_SLOT[slot]+off;}
__device__ __forceinline__ uint32_t bkf_direct_height(uint32_t x){return (x>>BKF_DIRECT_HEIGHT_SHIFT)&BKF_DIRECT_HEIGHT_MASK;}
__device__ __forceinline__ uint32_t bkf_direct_locator(uint32_t x){return x&BKF_LOC_MASK;}

__device__ __forceinline__ Count bkf_sum_high_preimages(
    uint32_t dest_code,uint32_t depth,uint32_t source_he,uint32_t source_bid,uint32_t source_low_loc
){
    Count sum=0;int s=int(depth);uint32_t low_slot=bucket_locator_owner(source_low_loc);
    uint32_t low_rank=bucket_locator_rank(source_low_loc);
    BucketPhysicalBlock sb=bkf_low_main(low_slot,source_bid);
#pragma unroll
    for(int pos=0;pos<HIGH_LUT_K;++pos){
        uint32_t v=(dest_code>>(2*pos))&3u;
        if(v==uint32_t(::L)){if(s==1)break;--s;}
        else if(v==uint32_t(R)){
            if(s==1){
                uint32_t z=3u<<(2*pos),src_code=(dest_code&~z)|(uint32_t(::L)<<(2*pos));
                uint32_t x=D_BKF_HIGH_DIRECT[gdx_ternary_key<HIGH_LUT_K>(src_code)];
                if(x!=BKF_DIRECT_INVALID&&bkf_direct_height(x)==source_he){
                    uint32_t hl=bkf_direct_locator(x);
                    if(bucket_locator_owner(hl)==D_BKF_FIXED_OWNER){
                        uint32_t hr=bucket_locator_rank(hl);
                        sum=gpu_direct_add(sum,bkf_ptr(low_slot,sb.off+Code(hr)*sb.cols+low_rank)[0]);
                    }
                }
            }
            ++s;
        }
    }
    return sum;
}

__device__ __forceinline__ Count bkf_sum_low_preimages(
    uint32_t dest_code,uint32_t depth,uint32_t source_hs,uint32_t source_bid,uint32_t source_high_loc
){
    Count sum=0;int s=int(depth);uint32_t high_slot=bucket_locator_owner(source_high_loc);
    uint32_t high_rank=bucket_locator_rank(source_high_loc);
    BucketPhysicalBlock sb=bkf_high_main(high_slot,source_bid);
#pragma unroll
    for(int pos=LOW_LUT_K-1;pos>=0;--pos){
        uint32_t v=(dest_code>>(2*pos))&3u;
        if(v==uint32_t(R)){if(s==1)break;--s;}
        else if(v==uint32_t(::L)){
            if(s==1){
                uint32_t z=3u<<(2*pos),src_code=(dest_code&~z)|(uint32_t(R)<<(2*pos));
                uint32_t x=D_BKF_LOW_DIRECT[gdx_ternary_key<LOW_LUT_K>(src_code)];
                if(x!=BKF_DIRECT_INVALID&&bkf_direct_height(x)==source_hs){
                    uint32_t ll=bkf_direct_locator(x);
                    if(bucket_locator_owner(ll)==D_BKF_FIXED_OWNER){
                        uint32_t lr=bucket_locator_rank(ll);
                        sum=gpu_direct_add(sum,bkf_ptr(high_slot,sb.off+Code(high_rank)*sb.cols+lr)[0]);
                    }
                }
            }
            ++s;
        }
    }
    return sum;
}

__global__ void bucket_low_orbit_kernel(int p){
    uint32_t bid=blockIdx.z;if(bid>=D_BKF_MAIN_NBLOCKS)return;
    uint32_t pi=uint32_t(LOW_LUT_K-p);size_t oi=size_t(pi)*D_BKF_LOW_PITCH+bid;
    uint32_t na=D_BKF_LOW_NN_OFF[oi],nb=D_BKF_LOW_NN_OFF[oi+1];
    uint32_t ra=D_BKF_LOW_NR_OFF[oi],rb=D_BKF_LOW_NR_OFF[oi+1];
    uint32_t la=D_BKF_LOW_NL_OFF[oi],lb=D_BKF_LOW_NL_OFF[oi+1];
    uint32_t n0=nb-na,n1=rb-ra,total=n0+n1+(lb-la);if(!total)return;
    for(uint32_t k=uint32_t(blockIdx.x)*blockDim.x+threadIdx.x;k<total;k+=uint32_t(gridDim.x)*blockDim.x){
        uint32_t kind;BucketOrbitOp op;
        if(k<n0){kind=CPU_ORBIT_NN;op=D_BKF_LOW_NN[na+k];}
        else if(k<n0+n1){kind=CPU_ORBIT_NR;op=D_BKF_LOW_NR[ra+k-n0];}
        else{kind=CPU_ORBIT_NL;op=D_BKF_LOW_NL[la+k-n0-n1];}
        uint32_t sl=bucket_orbit_src(op),jl=bucket_orbit_partner(op),dl=bucket_orbit_drop(op);
        uint32_t ss=bucket_locator_owner(sl),js=bucket_locator_owner(jl),ds=bucket_locator_owner(dl);
        BucketPhysicalBlock xb=bkf_low_main(ss,bid);
        if(!xb.valid||!xb.rows||!xb.cols)continue;
        uint32_t jbid=(p==LOW_LUT_K&&kind!=CPU_ORBIT_NN)?3u*uint32_t(xb.he)+(kind==CPU_ORBIT_NR?uint32_t(R):uint32_t(::L)):bid;
        BucketPhysicalBlock jb=bkf_low_main(js,jbid),db=bkf_low_block(ds,uint32_t(xb.he));
        uint32_t sr=bucket_locator_rank(sl),jr=bucket_locator_rank(jl),dr=bucket_locator_rank(dl);
        for(uint32_t hr=blockIdx.y;hr<xb.rows;hr+=gridDim.y){
            Count*ip=bkf_ptr(ss,xb.off+Code(hr)*xb.cols+sr);
            Count*jp=bkf_ptr(js,jb.off+Code(hr)*jb.cols+jr);
            Count*dp=bkf_ptr(ds,db.off+Code(hr)*db.cols+dr);
            Count c=*ip,old=*dp;
            if(kind==CPU_ORBIT_NN){*jp=gpu_direct_add(*jp,c);*ip=gpu_direct_add(c,old);*dp=0;}
            else{Count cc=*jp,all=gpu_direct_add(gpu_direct_add(c,cc),old);if(p==1){*ip=all;*jp=gpu_direct_add(c,cc);*dp=0;}else{*ip=all;*dp=c;}}
        }
    }
}

__global__ void bucket_high_orbit_kernel(int p){
    uint32_t bid=blockIdx.z;if(bid>=D_BKF_MAIN_NBLOCKS)return;
    uint32_t pi=uint32_t((TARGET_W-1)-p);size_t oi=size_t(pi)*D_BKF_HIGH_PITCH+bid;
    uint32_t na=D_BKF_HIGH_NN_OFF[oi],nb=D_BKF_HIGH_NN_OFF[oi+1];
    uint32_t ra=D_BKF_HIGH_NRNL_OFF[oi],rb=D_BKF_HIGH_NRNL_OFF[oi+1];
    uint32_t n0=nb-na,total=n0+(rb-ra);if(!total)return;
    for(uint32_t k=blockIdx.y;k<total;k+=gridDim.y){
        bool nn=k<n0;BucketOrbitOp op=nn?D_BKF_HIGH_NN[na+k]:D_BKF_HIGH_NRNL[ra+k-n0];
        uint32_t sl=bucket_orbit_src(op),jl=bucket_orbit_partner(op),dl=bucket_orbit_drop(op);
        uint32_t ss=bucket_locator_owner(sl),js=bucket_locator_owner(jl),ds=bucket_locator_owner(dl);
        BucketPhysicalBlock xb=bkf_high_main(ss,bid);if(!xb.valid||!xb.rows||!xb.cols)continue;
        uint32_t jbid=bid;
        if(p==LOW_LUT_K+1){uint32_t center=nn?uint32_t(R):uint32_t(N);int he=int(xb.hs)+(center==uint32_t(R)?1:0);jbid=uint32_t(3*he+int(center));}
        BucketPhysicalBlock jb=bkf_high_main(js,jbid),db=bkf_high_block(ds,uint32_t(xb.hs));
        uint32_t sr=bucket_locator_rank(sl),jr=bucket_locator_rank(jl),dr=bucket_locator_rank(dl);
        for(uint32_t lr=uint32_t(blockIdx.x)*blockDim.x+threadIdx.x;lr<xb.cols;lr+=uint32_t(gridDim.x)*blockDim.x){
            Count*ip=bkf_ptr(ss,xb.off+Code(sr)*xb.cols+lr);
            Count*jp=bkf_ptr(js,jb.off+Code(jr)*jb.cols+lr);
            Count*dp=bkf_ptr(ds,db.off+Code(dr)*db.cols+lr);
            Count c=*ip,old=*dp;
            if(nn){*jp=gpu_direct_add(*jp,c);*ip=gpu_direct_add(c,old);*dp=0;}
            else{Count cc=*jp;*ip=gpu_direct_add(gpu_direct_add(c,cc),old);*dp=c;}
        }
    }
}

__global__ void bucket_low_fused_closure_kernel(int p){
    uint32_t dbid=blockIdx.z;bool target_main=p==1;uint32_t nt=target_main?D_BKF_MAIN_NBLOCKS:D_BKF_BLOCK_NBLOCKS;if(dbid>=nt)return;
    uint32_t pi=uint32_t(LOW_LUT_K-p);size_t oi=size_t(pi)*D_BKF_LOW_FUSED_PITCH+dbid;
    uint32_t a=D_BKF_LOW_OFF[oi],b=D_BKF_LOW_OFF[oi+1];
    for(uint32_t q=a+uint32_t(blockIdx.x)*blockDim.x+threadIdx.x;q<b;q+=uint32_t(gridDim.x)*blockDim.x){
        BucketFusedDst rec=D_BKF_LOW_DST[q];uint32_t dslot=bucket_locator_owner(rec.dst_locator),dr=bucket_locator_rank(rec.dst_locator);
        BucketPhysicalBlock db=target_main?bkf_low_main(dslot,dbid):bkf_low_block(dslot,dbid);if(!db.valid||!db.rows||!db.cols)continue;
        uint32_t lc=rec.counts&0xffffu,cc=rec.counts>>16;
        for(uint32_t hr=blockIdx.y;hr<db.rows;hr+=gridDim.y){
            Count*dp=bkf_ptr(dslot,db.off+Code(hr)*db.cols+dr);Count sum=*dp;
            for(uint32_t e=rec.local_begin;e<rec.local_begin+lc;++e){uint32_t x=D_BKF_LOW_LOCAL_SRC[e],sl=bkf_src_locator(x),ss=bucket_locator_owner(sl);BucketPhysicalBlock sb=bkf_low_main(ss,bkf_src_block(x));sum=gpu_direct_add(sum,bkf_ptr(ss,sb.off+Code(hr)*sb.cols+bucket_locator_rank(sl))[0]);}
            if(cc){uint32_t dest_code=D_BKF_HIGH_CODES[D_BKF_HIGH_CODE_OFF[size_t(D_BKF_FIXED_OWNER)*D_BKF_CODE_PITCH+db.he]+hr];for(uint32_t e=rec.cross_begin;e<rec.cross_begin+cc;++e){uint32_t x=D_BKF_LOW_CROSS_OP[e],sl=bkf_src_locator(x);BucketPhysicalBlock sb=bkf_low_main(bucket_locator_owner(sl),bkf_src_block(x));sum=gpu_direct_add(sum,bkf_sum_high_preimages(dest_code,bkf_cross_depth(x),sb.he,bkf_src_block(x),sl));}}
            *dp=sum;
        }
    }
}

__global__ void bucket_high_fused_closure_kernel(int p){
    uint32_t dbid=blockIdx.z;if(dbid>=D_BKF_BLOCK_NBLOCKS)return;uint32_t pi=uint32_t((TARGET_W-1)-p);size_t oi=size_t(pi)*D_BKF_HIGH_FUSED_PITCH+dbid;
    uint32_t a=D_BKF_HIGH_OFF[oi],b=D_BKF_HIGH_OFF[oi+1];
    for(uint32_t q=a+blockIdx.y;q<b;q+=gridDim.y){
        BucketFusedDst rec=D_BKF_HIGH_DST[q];uint32_t dslot=bucket_locator_owner(rec.dst_locator),dr=bucket_locator_rank(rec.dst_locator);BucketPhysicalBlock db=bkf_high_block(dslot,dbid);if(!db.valid||!db.rows||!db.cols)continue;
        uint32_t lc=rec.counts&0xffffu,cc=rec.counts>>16;
        for(uint32_t lr=uint32_t(blockIdx.x)*blockDim.x+threadIdx.x;lr<db.cols;lr+=uint32_t(gridDim.x)*blockDim.x){
            Count*dp=bkf_ptr(dslot,db.off+Code(dr)*db.cols+lr);Count sum=*dp;
            for(uint32_t e=rec.local_begin;e<rec.local_begin+lc;++e){uint32_t x=D_BKF_HIGH_LOCAL_SRC[e],sl=bkf_src_locator(x),ss=bucket_locator_owner(sl);BucketPhysicalBlock sb=bkf_high_main(ss,bkf_src_block(x));sum=gpu_direct_add(sum,bkf_ptr(ss,sb.off+Code(bucket_locator_rank(sl))*sb.cols+lr)[0]);}
            if(cc){uint32_t dest_code=D_BKF_LOW_CODES[D_BKF_LOW_CODE_OFF[size_t(D_BKF_FIXED_OWNER)*D_BKF_CODE_PITCH+db.hs]+lr];for(uint32_t e=rec.cross_begin;e<rec.cross_begin+cc;++e){uint32_t x=D_BKF_HIGH_CROSS_OP[e],sl=bkf_src_locator(x);BucketPhysicalBlock sb=bkf_high_main(bucket_locator_owner(sl),bkf_src_block(x));sum=gpu_direct_add(sum,bkf_sum_low_preimages(dest_code,bkf_cross_depth(x),sb.hs,bkf_src_block(x),sl));}}
            *dp=sum;
        }
    }
}

struct BucketFusedDeviceTables{
    BucketPhysicalBlock *low_main=nullptr,*low_block=nullptr,*high_main=nullptr,*high_block=nullptr;
    BucketOrbitOp *low_nn=nullptr,*low_nr=nullptr,*low_nl=nullptr,*high_nn=nullptr,*high_nrnl=nullptr;
    uint32_t *low_nn_off=nullptr,*low_nr_off=nullptr,*low_nl_off=nullptr,*high_nn_off=nullptr,*high_nrnl_off=nullptr;
    BucketFusedDst *low_dst=nullptr,*high_dst=nullptr;uint32_t *low_off=nullptr,*high_off=nullptr;
    uint32_t *low_local_src=nullptr,*low_cross_op=nullptr,*high_local_src=nullptr,*high_cross_op=nullptr;
    uint32_t *high_codes=nullptr,*low_codes=nullptr,*high_code_off=nullptr,*low_code_off=nullptr,*high_direct=nullptr,*low_direct=nullptr;
    uint32_t main_nblocks=0,block_nblocks=0;

    template<class T>static void cp(T*&d,const std::vector<T>&s,const char*w){if(s.empty())return;ck(cudaMalloc(&d,s.size()*sizeof(T)),w);ck(cudaMemcpy(d,s.data(),s.size()*sizeof(T),cudaMemcpyHostToDevice),w);}

    void install_metadata(const StorageLayout&layout,const BucketOrbitStreamsHost&o,const BucketFusedHost&f){
        main_nblocks=uint32_t(layout.main_blocks.size());block_nblocks=uint32_t(layout.block_blocks.size());
        cp(low_nn,o.low_nn,"bkf low nn");cp(low_nr,o.low_nr,"bkf low nr");cp(low_nl,o.low_nl,"bkf low nl");cp(low_nn_off,o.low_nn_off,"bkf low nn off");cp(low_nr_off,o.low_nr_off,"bkf low nr off");cp(low_nl_off,o.low_nl_off,"bkf low nl off");
        cp(high_nn,o.high_nn,"bkf high nn");cp(high_nrnl,o.high_nrnl,"bkf high nrnl");cp(high_nn_off,o.high_nn_off,"bkf high nn off");cp(high_nrnl_off,o.high_nrnl_off,"bkf high nrnl off");
        cp(low_dst,f.low_dst,"bkf low dst");cp(low_off,f.low_off,"bkf low off");cp(low_local_src,f.low_local_src,"bkf low src");cp(low_cross_op,f.low_cross_op,"bkf low cross");
        cp(high_dst,f.high_dst,"bkf high dst");cp(high_off,f.high_off,"bkf high off");cp(high_local_src,f.high_local_src,"bkf high src");cp(high_cross_op,f.high_cross_op,"bkf high cross");
        cp(high_codes,f.high_codes,"bkf high codes");cp(low_codes,f.low_codes,"bkf low codes");cp(high_code_off,f.high_code_off,"bkf high code off");cp(low_code_off,f.low_code_off,"bkf low code off");cp(high_direct,f.high_direct,"bkf high direct");cp(low_direct,f.low_direct,"bkf low direct");
        uint32_t lp=o.low_nblocks+1,hp=o.high_nblocks+1;
        ck(cudaMemcpyToSymbol(D_BKF_MAIN_NBLOCKS,&main_nblocks,sizeof(main_nblocks)),"bkf main nblocks");ck(cudaMemcpyToSymbol(D_BKF_BLOCK_NBLOCKS,&block_nblocks,sizeof(block_nblocks)),"bkf block nblocks");
        ck(cudaMemcpyToSymbol(D_BKF_LOW_NN,&low_nn,sizeof(low_nn)),"bkf low nn ptr");ck(cudaMemcpyToSymbol(D_BKF_LOW_NR,&low_nr,sizeof(low_nr)),"bkf low nr ptr");ck(cudaMemcpyToSymbol(D_BKF_LOW_NL,&low_nl,sizeof(low_nl)),"bkf low nl ptr");ck(cudaMemcpyToSymbol(D_BKF_LOW_NN_OFF,&low_nn_off,sizeof(low_nn_off)),"bkf low nn off ptr");ck(cudaMemcpyToSymbol(D_BKF_LOW_NR_OFF,&low_nr_off,sizeof(low_nr_off)),"bkf low nr off ptr");ck(cudaMemcpyToSymbol(D_BKF_LOW_NL_OFF,&low_nl_off,sizeof(low_nl_off)),"bkf low nl off ptr");ck(cudaMemcpyToSymbol(D_BKF_LOW_PITCH,&lp,sizeof(lp)),"bkf low pitch");
        ck(cudaMemcpyToSymbol(D_BKF_HIGH_NN,&high_nn,sizeof(high_nn)),"bkf high nn ptr");ck(cudaMemcpyToSymbol(D_BKF_HIGH_NRNL,&high_nrnl,sizeof(high_nrnl)),"bkf high nrnl ptr");ck(cudaMemcpyToSymbol(D_BKF_HIGH_NN_OFF,&high_nn_off,sizeof(high_nn_off)),"bkf high nn off ptr");ck(cudaMemcpyToSymbol(D_BKF_HIGH_NRNL_OFF,&high_nrnl_off,sizeof(high_nrnl_off)),"bkf high nrnl off ptr");ck(cudaMemcpyToSymbol(D_BKF_HIGH_PITCH,&hp,sizeof(hp)),"bkf high pitch");
        ck(cudaMemcpyToSymbol(D_BKF_LOW_DST,&low_dst,sizeof(low_dst)),"bkf low dst ptr");ck(cudaMemcpyToSymbol(D_BKF_LOW_OFF,&low_off,sizeof(low_off)),"bkf low off ptr");ck(cudaMemcpyToSymbol(D_BKF_LOW_FUSED_PITCH,&f.low_pitch,sizeof(f.low_pitch)),"bkf low fused pitch");ck(cudaMemcpyToSymbol(D_BKF_LOW_LOCAL_SRC,&low_local_src,sizeof(low_local_src)),"bkf low src ptr");ck(cudaMemcpyToSymbol(D_BKF_LOW_CROSS_OP,&low_cross_op,sizeof(low_cross_op)),"bkf low cross ptr");
        ck(cudaMemcpyToSymbol(D_BKF_HIGH_DST,&high_dst,sizeof(high_dst)),"bkf high dst ptr");ck(cudaMemcpyToSymbol(D_BKF_HIGH_OFF,&high_off,sizeof(high_off)),"bkf high off ptr");ck(cudaMemcpyToSymbol(D_BKF_HIGH_FUSED_PITCH,&f.high_pitch,sizeof(f.high_pitch)),"bkf high fused pitch");ck(cudaMemcpyToSymbol(D_BKF_HIGH_LOCAL_SRC,&high_local_src,sizeof(high_local_src)),"bkf high src ptr");ck(cudaMemcpyToSymbol(D_BKF_HIGH_CROSS_OP,&high_cross_op,sizeof(high_cross_op)),"bkf high cross ptr");
        ck(cudaMemcpyToSymbol(D_BKF_HIGH_CODES,&high_codes,sizeof(high_codes)),"bkf high codes ptr");ck(cudaMemcpyToSymbol(D_BKF_LOW_CODES,&low_codes,sizeof(low_codes)),"bkf low codes ptr");ck(cudaMemcpyToSymbol(D_BKF_HIGH_CODE_OFF,&high_code_off,sizeof(high_code_off)),"bkf high code off ptr");ck(cudaMemcpyToSymbol(D_BKF_LOW_CODE_OFF,&low_code_off,sizeof(low_code_off)),"bkf low code off ptr");ck(cudaMemcpyToSymbol(D_BKF_CODE_PITCH,&f.code_pitch,sizeof(f.code_pitch)),"bkf code pitch");ck(cudaMemcpyToSymbol(D_BKF_HIGH_DIRECT,&high_direct,sizeof(high_direct)),"bkf high direct ptr");ck(cudaMemcpyToSymbol(D_BKF_LOW_DIRECT,&low_direct,sizeof(low_direct)),"bkf low direct ptr");
    }

    void bind_owner(uint32_t fixed,const BucketPhysicalLayoutHost&buckets,const std::array<Count*,BUCKET_NGPU>&slot){
        if(fixed>=BUCKET_NGPU){std::cerr<<"bucket fused fixed owner overflow\n";std::exit(207);}
        std::vector<BucketPhysicalBlock> lm,lb,hm,hb;lm.reserve(BUCKET_NGPU*main_nblocks);lb.reserve(BUCKET_NGPU*block_nblocks);hm.reserve(BUCKET_NGPU*main_nblocks);hb.reserve(BUCKET_NGPU*block_nblocks);
        for(uint32_t s=0;s<BUCKET_NGPU;++s){auto const&L=buckets.pair[fixed][s];lm.insert(lm.end(),L.main_blocks.begin(),L.main_blocks.end());lb.insert(lb.end(),L.block_blocks.begin(),L.block_blocks.end());auto const&H=buckets.pair[s][fixed];hm.insert(hm.end(),H.main_blocks.begin(),H.main_blocks.end());hb.insert(hb.end(),H.block_blocks.begin(),H.block_blocks.end());}
        auto replace=[&](BucketPhysicalBlock*&d,const std::vector<BucketPhysicalBlock>&v,const char*w){if(d)cudaFree(d);d=nullptr;cp(d,v,w);};
        replace(low_main,lm,"bkf low main view");replace(low_block,lb,"bkf low block view");replace(high_main,hm,"bkf high main view");replace(high_block,hb,"bkf high block view");
        ck(cudaMemcpyToSymbol(D_BKF_FIXED_OWNER,&fixed,sizeof(fixed)),"bkf fixed owner");ck(cudaMemcpyToSymbol(D_BKF_SLOT,slot.data(),sizeof(Count*)*BUCKET_NGPU),"bkf slots");ck(cudaMemcpyToSymbol(D_BKF_LOW_MAIN,&low_main,sizeof(low_main)),"bkf low main ptr");ck(cudaMemcpyToSymbol(D_BKF_LOW_BLOCK,&low_block,sizeof(low_block)),"bkf low block ptr");ck(cudaMemcpyToSymbol(D_BKF_HIGH_MAIN,&high_main,sizeof(high_main)),"bkf high main ptr");ck(cudaMemcpyToSymbol(D_BKF_HIGH_BLOCK,&high_block,sizeof(high_block)),"bkf high block ptr");
    }

    void release(){
#define BKF_FREE(x) do{if(x)cudaFree(x);x=nullptr;}while(0)
        BKF_FREE(low_main);BKF_FREE(low_block);BKF_FREE(high_main);BKF_FREE(high_block);BKF_FREE(low_nn);BKF_FREE(low_nr);BKF_FREE(low_nl);BKF_FREE(high_nn);BKF_FREE(high_nrnl);BKF_FREE(low_nn_off);BKF_FREE(low_nr_off);BKF_FREE(low_nl_off);BKF_FREE(high_nn_off);BKF_FREE(high_nrnl_off);BKF_FREE(low_dst);BKF_FREE(low_off);BKF_FREE(low_local_src);BKF_FREE(low_cross_op);BKF_FREE(high_dst);BKF_FREE(high_off);BKF_FREE(high_local_src);BKF_FREE(high_cross_op);BKF_FREE(high_codes);BKF_FREE(low_codes);BKF_FREE(high_code_off);BKF_FREE(low_code_off);BKF_FREE(high_direct);BKF_FREE(low_direct);
#undef BKF_FREE
    }
};

static void bucket_run_low_fused(const StorageLayout&layout,int threads=256,int grid_x=16,int grid_y=8){
    dim3 block(threads),grid(grid_x,grid_y,unsigned(layout.main_blocks.size()));
    for(int p=LOW_LUT_K;p>=1;--p){bucket_low_orbit_kernel<<<grid,block>>>(p);ck(cudaGetLastError(),"bucket low orbit");unsigned nt=p==1?unsigned(layout.main_blocks.size()):unsigned(layout.block_blocks.size());dim3 cg(grid_x,grid_y,nt);bucket_low_fused_closure_kernel<<<cg,block>>>(p);ck(cudaGetLastError(),"bucket low closure");}
    ck(cudaDeviceSynchronize(),"bucket low sync");
}
static void bucket_run_high_fused(const StorageLayout&layout,int threads=256,int grid_x=16,int grid_y=8){
    dim3 block(threads),og(grid_x,grid_y,unsigned(layout.main_blocks.size())),cg(grid_x,grid_y,unsigned(layout.block_blocks.size()));
    for(int p=TARGET_W-1;p>=LOW_LUT_K+1;--p){bucket_high_orbit_kernel<<<og,block>>>(p);ck(cudaGetLastError(),"bucket high orbit");bucket_high_fused_closure_kernel<<<cg,block>>>(p);ck(cudaGetLastError(),"bucket high closure");}
    ck(cudaDeviceSynchronize(),"bucket high sync");
}
