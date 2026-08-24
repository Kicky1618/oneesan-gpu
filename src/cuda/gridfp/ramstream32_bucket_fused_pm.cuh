#pragma once

#include "ramstream32_mod_accum.cuh"

// Experimental closure-only optimization for CRT primes close to 2^32.
// Orbit kernels keep the proven uint32 modular add path. Destination-gather
// closures accumulate raw uint32 sources in uint64_t and reduce once per
// destination cell with gpu_direct_pm_reduce_u64().

__device__ __forceinline__ uint64_t bkf_sum_high_preimages_u64(
    uint32_t dest_code,uint32_t depth,uint32_t source_he,
    uint32_t source_bid,uint32_t source_low_loc
){
    uint64_t sum=0;int s=int(depth);
    uint32_t low_slot=bkf_loc_owner(source_low_loc),low_rank=bkf_loc_rank(source_low_loc);
    BucketPhysicalBlock sb=bkf_high_main(low_slot,source_bid);
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
                    if(bkf_loc_owner(hl)==D_BKF_FIXED_OWNER){
                        uint32_t hr=bkf_loc_rank(hl);
                        sum+=uint64_t(bkf_ptr(low_slot,sb.off+Code(hr)*sb.cols+low_rank)[0]);
                    }
                }
            }
            ++s;
        }
    }
    return sum;
}

__device__ __forceinline__ uint64_t bkf_sum_low_preimages_u64(
    uint32_t dest_code,uint32_t depth,uint32_t source_hs,
    uint32_t source_bid,uint32_t source_high_loc
){
    uint64_t sum=0;int s=int(depth);
    uint32_t high_slot=bkf_loc_owner(source_high_loc),high_rank=bkf_loc_rank(source_high_loc);
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
                    if(bkf_loc_owner(ll)==D_BKF_FIXED_OWNER){
                        uint32_t lr=bkf_loc_rank(ll);
                        sum+=uint64_t(bkf_ptr(high_slot,sb.off+Code(high_rank)*sb.cols+lr)[0]);
                    }
                }
            }
            ++s;
        }
    }
    return sum;
}

__global__ void bucket_low_fused_closure_pm_kernel(int p){
    uint32_t dbid=blockIdx.z;bool target_main=p==1;
    uint32_t nt=target_main?D_BKF_MAIN_NBLOCKS:D_BKF_BLOCK_NBLOCKS;if(dbid>=nt)return;
    uint32_t pi=uint32_t(LOW_LUT_K-p);size_t oi=size_t(pi)*D_BKF_LOW_FUSED_PITCH+dbid;
    uint32_t a=D_BKF_LOW_OFF[oi],b=D_BKF_LOW_OFF[oi+1];
    for(uint32_t q=a+uint32_t(blockIdx.x)*blockDim.x+threadIdx.x;q<b;q+=uint32_t(gridDim.x)*blockDim.x){
        BucketFusedDst rec=D_BKF_LOW_DST[q];
        uint32_t dslot=bkf_loc_owner(rec.dst_locator),dr=bkf_loc_rank(rec.dst_locator);
        BucketPhysicalBlock db=target_main?bkf_low_main(dslot,dbid):bkf_low_block(dslot,dbid);
        if(!db.valid||!db.rows||!db.cols)continue;
        uint32_t lc=rec.counts&0xffffu,cc=rec.counts>>16;
        for(uint32_t hr=blockIdx.y;hr<db.rows;hr+=gridDim.y){
            Count*dp=bkf_ptr(dslot,db.off+Code(hr)*db.cols+dr);
            uint64_t sum=uint64_t(*dp);
            for(uint32_t e=rec.local_begin;e<rec.local_begin+lc;++e){
                uint32_t x=D_BKF_LOW_LOCAL_SRC[e],sl=bkf_src_locator(x),ss=bkf_loc_owner(sl);
                BucketPhysicalBlock sb=bkf_low_main(ss,bkf_src_block(x));
                sum+=uint64_t(bkf_ptr(ss,sb.off+Code(hr)*sb.cols+bkf_loc_rank(sl))[0]);
            }
            if(cc){
                uint32_t dest_code=D_BKF_HIGH_CODES[D_BKF_HIGH_CODE_OFF[size_t(D_BKF_FIXED_OWNER)*D_BKF_CODE_PITCH+db.he]+hr];
                for(uint32_t e=rec.cross_begin;e<rec.cross_begin+cc;++e){
                    uint32_t x=D_BKF_LOW_CROSS_OP[e],sl=bkf_src_locator(x);
                    BucketPhysicalBlock sb=bkf_low_main(bkf_loc_owner(sl),bkf_src_block(x));
                    sum+=bkf_sum_high_preimages_u64(dest_code,bkf_cross_depth(x),sb.he,bkf_src_block(x),sl);
                }
            }
            *dp=gpu_direct_pm_reduce_u64(sum);
        }
    }
}

__global__ void bucket_high_fused_closure_pm_kernel(int p){
    uint32_t dbid=blockIdx.z;if(dbid>=D_BKF_BLOCK_NBLOCKS)return;
    uint32_t pi=uint32_t((TARGET_W-1)-p);size_t oi=size_t(pi)*D_BKF_HIGH_FUSED_PITCH+dbid;
    uint32_t a=D_BKF_HIGH_OFF[oi],b=D_BKF_HIGH_OFF[oi+1];
    for(uint32_t q=a+blockIdx.y;q<b;q+=gridDim.y){
        BucketFusedDst rec=D_BKF_HIGH_DST[q];
        uint32_t dslot=bkf_loc_owner(rec.dst_locator),dr=bkf_loc_rank(rec.dst_locator);
        BucketPhysicalBlock db=bkf_high_block(dslot,dbid);if(!db.valid||!db.rows||!db.cols)continue;
        uint32_t lc=rec.counts&0xffffu,cc=rec.counts>>16;
        for(uint32_t lr=uint32_t(blockIdx.x)*blockDim.x+threadIdx.x;lr<db.cols;lr+=uint32_t(gridDim.x)*blockDim.x){
            Count*dp=bkf_ptr(dslot,db.off+Code(dr)*db.cols+lr);
            uint64_t sum=uint64_t(*dp);
            for(uint32_t e=rec.local_begin;e<rec.local_begin+lc;++e){
                uint32_t x=D_BKF_HIGH_LOCAL_SRC[e],sl=bkf_src_locator(x),ss=bkf_loc_owner(sl);
                BucketPhysicalBlock sb=bkf_high_main(ss,bkf_src_block(x));
                sum+=uint64_t(bkf_ptr(ss,sb.off+Code(bkf_loc_rank(sl))*sb.cols+lr)[0]);
            }
            if(cc){
                uint32_t dest_code=D_BKF_LOW_CODES[D_BKF_LOW_CODE_OFF[size_t(D_BKF_FIXED_OWNER)*D_BKF_CODE_PITCH+db.hs]+lr];
                for(uint32_t e=rec.cross_begin;e<rec.cross_begin+cc;++e){
                    uint32_t x=D_BKF_HIGH_CROSS_OP[e],sl=bkf_src_locator(x);
                    BucketPhysicalBlock sb=bkf_high_main(bkf_loc_owner(sl),bkf_src_block(x));
                    sum+=bkf_sum_low_preimages_u64(dest_code,bkf_cross_depth(x),sb.hs,bkf_src_block(x),sl);
                }
            }
            *dp=gpu_direct_pm_reduce_u64(sum);
        }
    }
}

static void bucket_run_low_fused_pm(const StorageLayout&layout,int threads=256,int grid_x=16,int grid_y=8){
    dim3 block(threads),grid(grid_x,grid_y,unsigned(layout.main_blocks.size()));
    for(int p=LOW_LUT_K;p>=1;--p){
        bucket_low_orbit_kernel<<<grid,block>>>(p);ck(cudaGetLastError(),"bucket low orbit pm");
        unsigned nt=p==1?unsigned(layout.main_blocks.size()):unsigned(layout.block_blocks.size());
        dim3 cg(grid_x,grid_y,nt);bucket_low_fused_closure_pm_kernel<<<cg,block>>>(p);ck(cudaGetLastError(),"bucket low closure pm");
    }
    ck(cudaDeviceSynchronize(),"bucket low pm sync");
}

static void bucket_run_high_fused_pm(const StorageLayout&layout,int threads=256,int grid_x=16,int grid_y=8){
    dim3 block(threads),og(grid_x,grid_y,unsigned(layout.main_blocks.size())),cg(grid_x,grid_y,unsigned(layout.block_blocks.size()));
    for(int p=TARGET_W-1;p>=LOW_LUT_K+1;--p){
        bucket_high_orbit_kernel<<<og,block>>>(p);ck(cudaGetLastError(),"bucket high orbit pm");
        bucket_high_fused_closure_pm_kernel<<<cg,block>>>(p);ck(cudaGetLastError(),"bucket high closure pm");
    }
    ck(cudaDeviceSynchronize(),"bucket high pm sync");
}

static void bucket_launch_low_fused_pm(const StorageLayout&layout,int threads=256,int grid_x=16,int grid_y=8){
    dim3 block(threads),grid(grid_x,grid_y,unsigned(layout.main_blocks.size()));
    for(int p=LOW_LUT_K;p>=1;--p){
        bucket_low_orbit_kernel<<<grid,block>>>(p);ck(cudaGetLastError(),"bucket low orbit pm async");
        unsigned nt=p==1?unsigned(layout.main_blocks.size()):unsigned(layout.block_blocks.size());
        dim3 cg(grid_x,grid_y,nt);bucket_low_fused_closure_pm_kernel<<<cg,block>>>(p);ck(cudaGetLastError(),"bucket low closure pm async");
    }
}

static void bucket_launch_high_fused_pm(const StorageLayout&layout,int threads=256,int grid_x=16,int grid_y=8){
    dim3 block(threads),og(grid_x,grid_y,unsigned(layout.main_blocks.size())),cg(grid_x,grid_y,unsigned(layout.block_blocks.size()));
    for(int p=TARGET_W-1;p>=LOW_LUT_K+1;--p){
        bucket_high_orbit_kernel<<<og,block>>>(p);ck(cudaGetLastError(),"bucket high orbit pm async");
        bucket_high_fused_closure_pm_kernel<<<cg,block>>>(p);ck(cudaGetLastError(),"bucket high closure pm async");
    }
}
