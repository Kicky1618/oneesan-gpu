#pragma once

#include "ramstream32_bucket_orbit_closure_pattern10_depth8.cuh"

// HIGH-window kernels map one orbit operation to blockIdx.y and spread its LOW
// columns across threadIdx.x.  All x-threads therefore used to reload/decode
// the same orbit word and three BucketPhysicalBlocks.  Hoist that invariant
// context, together with the already-shared closure plan, into one small shared
// object built by thread 0 once per orbit operation.
struct P10D8HighCtx {
    BucketPhysicalBlock xb{},jb{},db{};
    BkczPlan plan{};
    uint32_t ss=0,js=0,ds=0,sr=0,jr=0,dr=0;
    uint32_t n0=0,n1=0,total=0;
    uint8_t kind=0,valid=0,pad0=0,pad1=0;
};

__global__ void bucket_high_orbit_closure_pattern10_depth8_highctx_kernel(int p){
    uint32_t bid=blockIdx.z;if(bid>=D_BKF_MAIN_NBLOCKS)return;uint32_t pi=uint32_t((TARGET_W-1)-p),oi=uint32_t(size_t(pi)*D_BKF_HIGH_PITCH+bid);__shared__ P10D8HighCtx c;
    if(threadIdx.x==0){uint32_t na=D_BKF_HIGH_NN_OFF[oi],nb=D_BKF_HIGH_NN_OFF[oi+1],ra=D_BKF_HIGH_NRNL_OFF[oi],rb=D_BKF_HIGH_NRNL_OFF[oi+1];c.n0=nb-na;c.n1=ra;c.total=c.n0+(rb-ra);}__syncthreads();
    for(uint32_t k=blockIdx.y;k<c.total;k+=gridDim.y){
        if(threadIdx.x==0){
            c.valid=0;bool nn=k<c.n0;uint32_t qi=nn?(D_BKF_HIGH_NN_OFF[oi]+k):(c.n1+k-c.n0);BucketOrbitOp op=nn?D_BKF_HIGH_NN[qi]:D_BKF_HIGH_NRNL[qi];uint8_t dep=nn?D_P10D8_F_HIGH_NN[qi]:D_P10D8_F_HIGH_NRNL[qi];uint32_t sl=bkf_orbit_src(op),jl=bkf_orbit_partner(op),dl=bkf_orbit_drop(op);c.ss=bkf_loc_owner(sl);c.js=bkf_loc_owner(jl);c.ds=bkf_loc_owner(dl);c.xb=bkf_high_main(c.ss,bid);
            if(c.xb.valid&&c.xb.rows&&c.xb.cols){uint32_t jbid=bid;if(p==LOW_LUT_K+1){uint32_t center=nn?uint32_t(R):uint32_t(N);int he=int(c.xb.hs)+(center==uint32_t(R)?1:0);jbid=uint32_t(3*he+int(center));}c.jb=bkf_high_main(c.js,jbid);c.db=bkf_high_block(c.ds,uint32_t(c.xb.hs));c.sr=bkf_loc_rank(sl);c.jr=bkf_loc_rank(jl);c.dr=bkf_loc_rank(dl);c.kind=uint8_t(nn?CPU_ORBIT_NN:CPU_ORBIT_NR);c.plan=bkcpd8_forward_high(bkcp10_id(op),dep,dl,c.db,p);c.valid=1;}
        }
        __syncthreads();
        if(c.valid){for(uint32_t lr=uint32_t(blockIdx.x)*blockDim.x+threadIdx.x;lr<c.xb.cols;lr+=uint32_t(gridDim.x)*blockDim.x){Count*ip=bkf_ptr(c.ss,c.xb.off+Code(c.sr)*c.xb.cols+lr),*jp=bkf_ptr(c.js,c.jb.off+Code(c.jr)*c.jb.cols+lr),*dp=bkf_ptr(c.ds,c.db.off+Code(c.dr)*c.db.cols+lr);Count x=*ip,old=*dp,extra=bkcz_high_plan_sum(c.plan,c.db,lr);if(c.kind==CPU_ORBIT_NN){*jp=gpu_direct_add(*jp,x);*ip=gpu_direct_add(x,old);*dp=extra;}else{Count y=*jp;*ip=gpu_direct_add(gpu_direct_add(x,y),old);*dp=gpu_direct_add(x,extra);}}}
        __syncthreads();
    }
}

__global__ void bucket_reverse_high_pattern10_depth8_highctx_kernel(int p){
    uint32_t bid=blockIdx.z;if(bid>=D_BKF_MAIN_NBLOCKS)return;uint32_t pi=uint32_t(p-(LOW_LUT_K+1)),oi=uint32_t(size_t(pi)*D_RS54_PITCH+bid);bool edge=p==TARGET_W-1;__shared__ P10D8HighCtx c;
    if(threadIdx.x==0){uint32_t na=D_RS54_HIGH_NN_OFF[oi],nb=D_RS54_HIGH_NN_OFF[oi+1],ra=D_RS54_HIGH_NR_OFF[oi],rb=D_RS54_HIGH_NR_OFF[oi+1],la=D_RS54_HIGH_NL_OFF[oi],lb=D_RS54_HIGH_NL_OFF[oi+1];c.n0=nb-na;c.n1=rb-ra;c.total=c.n0+c.n1+(lb-la);}__syncthreads();
    for(uint32_t k=blockIdx.y;k<c.total;k+=gridDim.y){
        if(threadIdx.x==0){
            c.valid=0;uint32_t qi=0,kind=0;BucketOrbitOp op;uint8_t dep;if(k<c.n0){kind=CPU_ORBIT_NN;qi=D_RS54_HIGH_NN_OFF[oi]+k;op=D_RS54_HIGH_NN[qi];dep=D_P10D8_R_HIGH_NN[qi];}else if(k<c.n0+c.n1){kind=CPU_ORBIT_NR;qi=D_RS54_HIGH_NR_OFF[oi]+k-c.n0;op=D_RS54_HIGH_NR[qi];dep=D_P10D8_R_HIGH_NR[qi];}else{kind=CPU_ORBIT_NL;qi=D_RS54_HIGH_NL_OFF[oi]+k-c.n0-c.n1;op=D_RS54_HIGH_NL[qi];dep=D_P10D8_R_HIGH_NL[qi];}uint32_t sl=bkf_orbit_src(op),jl=bkf_orbit_partner(op),dl=bkf_orbit_drop(op);c.ss=bkf_loc_owner(sl);c.js=bkf_loc_owner(jl);c.ds=bkf_loc_owner(dl);c.xb=bkf_high_main(c.ss,bid);
            if(c.xb.valid&&c.xb.rows&&c.xb.cols){c.jb=bkf_high_main(c.js,bkcp10_reverse_high_jblock(bid,c.xb,p,kind));c.db=bkf_high_block(c.ds,uint32_t(c.xb.hs));c.sr=bkf_loc_rank(sl);c.jr=bkf_loc_rank(jl);c.dr=bkf_loc_rank(dl);c.kind=uint8_t(kind);c.plan=edge?bkcpd8_reverse_high(bkcp10_id(op),dep,sl,c.xb,p,true):bkcpd8_reverse_high(bkcp10_id(op),dep,dl,c.db,p,false);c.valid=1;}
        }
        __syncthreads();
        if(c.valid){for(uint32_t lr=uint32_t(blockIdx.x)*blockDim.x+threadIdx.x;lr<c.xb.cols;lr+=uint32_t(gridDim.x)*blockDim.x){Count*ip=bkf_ptr(c.ss,c.xb.off+Code(c.sr)*c.xb.cols+lr),*jp=bkf_ptr(c.js,c.jb.off+Code(c.jr)*c.jb.cols+lr),*dp=bkf_ptr(c.ds,c.db.off+Code(c.dr)*c.db.cols+lr);Count x=*ip,old=*dp,extra=bkcz_high_plan_sum(c.plan,edge?c.xb:c.db,lr);if(c.kind==CPU_ORBIT_NN){*jp=gpu_direct_add(*jp,x);*ip=gpu_direct_add(gpu_direct_add(x,old),edge?extra:0);*dp=edge?0:extra;}else{Count y=*jp;*ip=gpu_direct_add(gpu_direct_add(x,y),old);if(edge){*jp=gpu_direct_add(x,y);*dp=0;}else *dp=gpu_direct_add(x,extra);}}}
        __syncthreads();
    }
}

static void bucket_launch_high_orbit_closure_pattern10_depth8_highctx(const StorageLayout&layout,int threads=256,int gx=16,int gy=8){dim3 block(threads),grid(gx,gy,unsigned(layout.main_blocks.size()));for(int p=TARGET_W-1;p>=LOW_LUT_K+1;--p){bucket_high_orbit_closure_pattern10_depth8_highctx_kernel<<<grid,block>>>(p);ck(cudaGetLastError(),"bucket high pattern10 depth8 highctx");}}
static void bucket_launch_reverse_high_pattern10_depth8_highctx(const StorageLayout&layout,int threads=256,int gx=16,int gy=8){dim3 block(threads),grid(gx,gy,unsigned(layout.main_blocks.size()));for(int p=LOW_LUT_K+1;p<TARGET_W;++p){bucket_reverse_high_pattern10_depth8_highctx_kernel<<<grid,block>>>(p);ck(cudaGetLastError(),"bucket reverse high pattern10 depth8 highctx");}}
