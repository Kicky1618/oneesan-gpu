#pragma once

#include "ramstream32_bucket_reverse_fused.cuh"
#include "ramstream32_bucket_fused_pm.cuh"

// Pseudo-Mersenne variant of the atomic-free reverse closure kernels.  The
// repository CRT primes are p=2^32-c with small c, so all ordinary and CROSS
// contributors are accumulated raw in uint64_t and reduced once per
// destination cell.  Reverse orbit stays on the proven uint32 add path.

__global__ void bucket_reverse_low_fused_closure_pm_kernel(int p) {
    uint32_t dbid=blockIdx.z;if(dbid>=D_BKF_BLOCK_NBLOCKS)return;
    uint32_t pi=uint32_t(p-1);size_t oi=size_t(pi)*D_RBF_LOW_PITCH+dbid;
    uint32_t a=D_RBF_LOW_OFF[oi],b=D_RBF_LOW_OFF[oi+1];
    for(uint32_t q=a+uint32_t(blockIdx.x)*blockDim.x+threadIdx.x;q<b;q+=uint32_t(gridDim.x)*blockDim.x){
        BucketFusedDst rec=D_RBF_LOW_DST[q];uint32_t dslot=bkf_loc_owner(rec.dst_locator),dr=bkf_loc_rank(rec.dst_locator);
        BucketPhysicalBlock db=bkf_low_block(dslot,dbid);if(!db.valid||!db.rows||!db.cols)continue;
        uint32_t lc=rec.counts&0xffffu,cc=rec.counts>>16;
        for(uint32_t hr=blockIdx.y;hr<db.rows;hr+=gridDim.y){
            Count*dp=bkf_ptr(dslot,db.off+Code(hr)*db.cols+dr);uint64_t sum=uint64_t(*dp);
            for(uint32_t e=rec.local_begin;e<rec.local_begin+lc;++e){
                uint32_t x=D_RBF_LOW_LOCAL_SRC[e],sl=bkf_src_locator(x),ss=bkf_loc_owner(sl);
                BucketPhysicalBlock sb=bkf_low_main(ss,bkf_src_block(x));
                sum+=uint64_t(bkf_ptr(ss,sb.off+Code(hr)*sb.cols+bkf_loc_rank(sl))[0]);
            }
            if(cc){
                uint32_t dest_code=D_BKF_HIGH_CODES[D_BKF_HIGH_CODE_OFF[size_t(D_BKF_FIXED_OWNER)*D_BKF_CODE_PITCH+db.he]+hr];
                for(uint32_t e=rec.cross_begin;e<rec.cross_begin+cc;++e){
                    uint32_t x=D_RBF_LOW_CROSS_OP[e],sl=bkf_src_locator(x);uint32_t sbid=bkf_src_block(x);
                    BucketPhysicalBlock sb=bkf_low_main(bkf_loc_owner(sl),sbid);
                    sum+=bkf_sum_high_preimages_u64(dest_code,bkf_cross_depth(x),sb.he,sbid,sl);
                }
            }
            *dp=gpu_direct_pm_reduce_u64(sum);
        }
    }
}

__global__ void bucket_reverse_high_fused_closure_pm_kernel(int p) {
    bool target_main=p==TARGET_W-1;uint32_t dbid=blockIdx.z;
    uint32_t nt=target_main?D_BKF_MAIN_NBLOCKS:D_BKF_BLOCK_NBLOCKS;if(dbid>=nt)return;
    uint32_t pi=uint32_t(p-(LOW_LUT_K+1));size_t oi=size_t(pi)*D_RBF_HIGH_PITCH+dbid;
    uint32_t a=D_RBF_HIGH_OFF[oi],b=D_RBF_HIGH_OFF[oi+1];
    for(uint32_t q=a+blockIdx.y;q<b;q+=gridDim.y){
        BucketFusedDst rec=D_RBF_HIGH_DST[q];uint32_t dslot=bkf_loc_owner(rec.dst_locator),dr=bkf_loc_rank(rec.dst_locator);
        BucketPhysicalBlock db=target_main?bkf_high_main(dslot,dbid):bkf_high_block(dslot,dbid);if(!db.valid||!db.rows||!db.cols)continue;
        uint32_t lc=rec.counts&0xffffu,cc=rec.counts>>16;
        for(uint32_t lr=uint32_t(blockIdx.x)*blockDim.x+threadIdx.x;lr<db.cols;lr+=uint32_t(gridDim.x)*blockDim.x){
            Count*dp=bkf_ptr(dslot,db.off+Code(dr)*db.cols+lr);uint64_t sum=uint64_t(*dp);
            for(uint32_t e=rec.local_begin;e<rec.local_begin+lc;++e){
                uint32_t x=D_RBF_HIGH_LOCAL_SRC[e],sl=bkf_src_locator(x),ss=bkf_loc_owner(sl);
                BucketPhysicalBlock sb=bkf_high_main(ss,bkf_src_block(x));
                sum+=uint64_t(bkf_ptr(ss,sb.off+Code(bkf_loc_rank(sl))*sb.cols+lr)[0]);
            }
            if(cc){
                uint32_t dest_code=D_BKF_LOW_CODES[D_BKF_LOW_CODE_OFF[size_t(D_BKF_FIXED_OWNER)*D_BKF_CODE_PITCH+db.hs]+lr];
                for(uint32_t e=rec.cross_begin;e<rec.cross_begin+cc;++e){
                    uint32_t x=D_RBF_HIGH_CROSS_OP[e],sl=bkf_src_locator(x);uint32_t sbid=bkf_src_block(x);
                    BucketPhysicalBlock sb=bkf_high_main(bkf_loc_owner(sl),sbid);
                    sum+=bkf_sum_low_preimages_u64(dest_code,bkf_cross_depth(x),sb.hs,sbid,sl);
                }
            }
            *dp=gpu_direct_pm_reduce_u64(sum);
        }
    }
}

static void bucket_launch_reverse_low_fused_pm(
    const StorageLayout&layout,int threads=256,int grid_x=16,int grid_y=8
){
    dim3 block(threads),og(grid_x,grid_y,unsigned(layout.main_blocks.size()));
    dim3 cg(grid_x,grid_y,unsigned(layout.block_blocks.size()));
    for(int p=1;p<=LOW_LUT_K;++p){
        bucket_reverse_low_orbit_kernel<<<og,block>>>(p);ck(cudaGetLastError(),"reverse fused low orbit pm");
        bucket_reverse_low_fused_closure_pm_kernel<<<cg,block>>>(p);ck(cudaGetLastError(),"reverse fused low closure pm");
    }
}
static void bucket_launch_reverse_high_fused_pm(
    const StorageLayout&layout,int threads=256,int grid_x=16,int grid_y=8
){
    dim3 block(threads),og(grid_x,grid_y,unsigned(layout.main_blocks.size()));
    for(int p=LOW_LUT_K+1;p<TARGET_W;++p){
        bucket_reverse_high_orbit_kernel<<<og,block>>>(p);ck(cudaGetLastError(),"reverse fused high orbit pm");
        unsigned nt=p==TARGET_W-1?unsigned(layout.main_blocks.size()):unsigned(layout.block_blocks.size());
        dim3 cg(grid_x,grid_y,nt);
        bucket_reverse_high_fused_closure_pm_kernel<<<cg,block>>>(p);ck(cudaGetLastError(),"reverse fused high closure pm");
    }
}
