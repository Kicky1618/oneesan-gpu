#pragma once
#include "ramstream32_bucket_orbit_closure_zero.cuh"

// HIGH windows map one orbit operation to blockIdx.y while x threads sweep
// contiguous LOW columns. The closure inverse plan is therefore identical for
// every thread in the block. Build it once in thread 0 and keep it in shared
// memory instead of reconstructing the same topology blockDim.x times.

__global__ void bucket_high_orbit_closure_zero_shared_kernel(int p){
    uint32_t bid=blockIdx.z;if(bid>=D_BKF_MAIN_NBLOCKS)return;
    uint32_t pi=uint32_t((TARGET_W-1)-p);size_t oi=size_t(pi)*D_BKF_HIGH_PITCH+bid;
    uint32_t na=D_BKF_HIGH_NN_OFF[oi],nb=D_BKF_HIGH_NN_OFF[oi+1],ra=D_BKF_HIGH_NRNL_OFF[oi],rb=D_BKF_HIGH_NRNL_OFF[oi+1];
    uint32_t n0=nb-na,total=n0+(rb-ra);if(!total)return;
    __shared__ BkczPlan plan;
    for(uint32_t k=blockIdx.y;k<total;k+=gridDim.y){
        bool nn=k<n0;uint32_t qi=nn?na+k:ra+k-n0;BucketOrbitOp op=nn?D_BKF_HIGH_NN[qi]:D_BKF_HIGH_NRNL[qi];
        uint32_t sl=bkf_orbit_src(op),jl=bkf_orbit_partner(op),dl=bkf_orbit_drop(op),ss=bkf_loc_owner(sl),js=bkf_loc_owner(jl),ds=bkf_loc_owner(dl);
        BucketPhysicalBlock xb=bkf_high_main(ss,bid);if(!xb.valid||!xb.rows||!xb.cols)continue;
        uint32_t jbid=bid;if(p==LOW_LUT_K+1){uint32_t center=nn?uint32_t(R):uint32_t(N);int he=int(xb.hs)+(center==uint32_t(R)?1:0);jbid=uint32_t(3*he+int(center));}
        BucketPhysicalBlock jb=bkf_high_main(js,jbid),db=bkf_high_block(ds,uint32_t(xb.hs));
        uint32_t sr=bkf_loc_rank(sl),jr=bkf_loc_rank(jl),dr=bkf_loc_rank(dl);
        if(threadIdx.x==0)plan=bkcz_forward_high_plan(dl,db,p);
        __syncthreads();
        for(uint32_t lr=uint32_t(blockIdx.x)*blockDim.x+threadIdx.x;lr<xb.cols;lr+=uint32_t(gridDim.x)*blockDim.x){
            Count*ip=bkf_ptr(ss,xb.off+Code(sr)*xb.cols+lr),*jp=bkf_ptr(js,jb.off+Code(jr)*jb.cols+lr),*dp=bkf_ptr(ds,db.off+Code(dr)*db.cols+lr);
            Count c=*ip,old=*dp,extra=bkcz_high_plan_sum(plan,db,lr);
            if(nn){*jp=gpu_direct_add(*jp,c);*ip=gpu_direct_add(c,old);*dp=extra;}
            else{Count cc=*jp;*ip=gpu_direct_add(gpu_direct_add(c,cc),old);*dp=gpu_direct_add(c,extra);}
        }
        __syncthreads();
    }
}

__global__ void bucket_reverse_high_split54_zero_shared_kernel(int p){
    uint32_t bid=blockIdx.z;if(bid>=D_BKF_MAIN_NBLOCKS)return;
    uint32_t pi=uint32_t(p-(LOW_LUT_K+1));size_t oi=size_t(pi)*D_RS54_PITCH+bid;
    uint32_t na=D_RS54_HIGH_NN_OFF[oi],nb=D_RS54_HIGH_NN_OFF[oi+1],ra=D_RS54_HIGH_NR_OFF[oi],rb=D_RS54_HIGH_NR_OFF[oi+1],la=D_RS54_HIGH_NL_OFF[oi],lb=D_RS54_HIGH_NL_OFF[oi+1];
    uint32_t n0=nb-na,n1=rb-ra,total=n0+n1+(lb-la);if(!total)return;bool edge=p==TARGET_W-1;
    __shared__ BkczPlan plan;
    for(uint32_t k=blockIdx.y;k<total;k+=gridDim.y){
        uint32_t kind;BucketOrbitOp op;
        if(k<n0){kind=CPU_ORBIT_NN;op=D_RS54_HIGH_NN[na+k];}
        else if(k<n0+n1){kind=CPU_ORBIT_NR;op=D_RS54_HIGH_NR[ra+k-n0];}
        else{kind=CPU_ORBIT_NL;op=D_RS54_HIGH_NL[la+k-n0-n1];}
        uint32_t sl=bkf_orbit_src(op),jl=bkf_orbit_partner(op),dl=bkf_orbit_drop(op),ss=bkf_loc_owner(sl),js=bkf_loc_owner(jl),ds=bkf_loc_owner(dl);
        BucketPhysicalBlock xb=bkf_high_main(ss,bid),jb=bkf_high_main(js,rs54_high_jblock(bid,xb,p,kind)),db=bkf_high_block(ds,uint32_t(xb.hs));
        if(threadIdx.x==0){plan=BkczPlan{};if(edge){if(kind==CPU_ORBIT_NN)plan=bkcz_reverse_high_plan(sl,xb,p,true);}else plan=bkcz_reverse_high_plan(dl,db,p,false);}
        __syncthreads();
        for(uint32_t lr=uint32_t(blockIdx.x)*blockDim.x+threadIdx.x;lr<xb.cols;lr+=uint32_t(gridDim.x)*blockDim.x){
            Count*ip=bkf_ptr(ss,xb.off+Code(bkf_loc_rank(sl))*xb.cols+lr),*jp=bkf_ptr(js,jb.off+Code(bkf_loc_rank(jl))*jb.cols+lr),*dp=bkf_ptr(ds,db.off+Code(bkf_loc_rank(dl))*db.cols+lr);
            Count c=*ip,old=*dp,extra=bkcz_high_plan_sum(plan,edge?xb:db,lr);
            if(kind==CPU_ORBIT_NN){*jp=gpu_direct_add(*jp,c);*ip=gpu_direct_add(gpu_direct_add(c,old),edge?extra:0);*dp=edge?0:extra;}
            else{Count cc=*jp;*ip=gpu_direct_add(gpu_direct_add(c,cc),old);if(edge){*jp=gpu_direct_add(c,cc);*dp=0;}else *dp=gpu_direct_add(c,extra);}
        }
        __syncthreads();
    }
}

static void bucket_launch_high_orbit_closure_zero_shared(const StorageLayout&layout,int threads=256,int gx=16,int gy=8){
    dim3 block(threads),grid(gx,gy,unsigned(layout.main_blocks.size()));
    for(int p=TARGET_W-1;p>=LOW_LUT_K+1;--p){bucket_high_orbit_closure_zero_shared_kernel<<<grid,block>>>(p);ck(cudaGetLastError(),"bucket high closure zero shared");}
}
static void bucket_launch_reverse_high_split54_zero_shared(const StorageLayout&layout,int threads=256,int gx=16,int gy=8){
    dim3 block(threads),grid(gx,gy,unsigned(layout.main_blocks.size()));
    for(int p=LOW_LUT_K+1;p<TARGET_W;++p){bucket_reverse_high_split54_zero_shared_kernel<<<grid,block>>>(p);ck(cudaGetLastError(),"bucket reverse high split54 zero shared");}
}

static void bucket_enqueue_high_orbit_closure_zero_shared(const StorageLayout&layout,cudaStream_t stream,int threads=256,int gx=16,int gy=8){
    dim3 block(threads),grid(gx,gy,unsigned(layout.main_blocks.size()));
    for(int p=TARGET_W-1;p>=LOW_LUT_K+1;--p){bucket_high_orbit_closure_zero_shared_kernel<<<grid,block,0,stream>>>(p);ck(cudaGetLastError(),"bucket high closure zero shared stream");}
}
static void bucket_enqueue_reverse_high_split54_zero_shared(const StorageLayout&layout,cudaStream_t stream,int threads=256,int gx=16,int gy=8){
    dim3 block(threads),grid(gx,gy,unsigned(layout.main_blocks.size()));
    for(int p=LOW_LUT_K+1;p<TARGET_W;++p){bucket_reverse_high_split54_zero_shared_kernel<<<grid,block,0,stream>>>(p);ck(cudaGetLastError(),"bucket reverse high split54 zero shared stream");}
}
