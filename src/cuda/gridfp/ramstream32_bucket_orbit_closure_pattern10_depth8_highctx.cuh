#pragma once

#include "ramstream32_bucket_orbit_closure_pattern10_depth8.cuh"

#ifndef P10D8_HIGH_DEPTH_LOAD
#define P10D8_HIGH_DEPTH_LOAD(ptr,q) ((ptr)[(q)])
#define P10D8_HIGH_DEPTH_LOAD_LOCAL 1
#endif
#ifndef P10D8_HIGHCTX_RESOLVE
#define P10D8_HIGHCTX_RESOLVE 0
#endif
static_assert(P10D8_HIGHCTX_RESOLVE==0||P10D8_HIGHCTX_RESOLVE==1,
              "P10D8_HIGHCTX_RESOLVE must be 0 or 1");

// HIGH-window kernels map one orbit operation to blockIdx.y and spread its LOW
// columns across threadIdx.x. All x-threads share the same orbit/addressing
// context and closure plan, so thread 0 constructs them once per operation.
//
// In resolved mode thread 0 also turns every local closure source into a row
// base pointer. The hot x-threads then load base[lr] directly instead of
// repeating locator decode, physical-block lookup and slot-pointer arithmetic
// for every source in every column.
struct P10D8HighCtx {
    BucketPhysicalBlock xb{},jb{},db{};
    BkczPlan plan{};
    uint32_t ss=0,js=0,ds=0,sr=0,jr=0,dr=0;
    uint32_t n0=0,n1=0,total=0;
    uint8_t kind=0,valid=0,pad0=0,pad1=0;
#if P10D8_HIGHCTX_RESOLVE
    Count* local_base[BKCZ_MAX_LOCAL]{};
    Count* cross_base=nullptr;
    uint8_t local_n=0;
#endif
};

#if P10D8_HIGHCTX_RESOLVE
__device__ __forceinline__ void p10d8_highctx_resolve_plan(P10D8HighCtx& c){
    uint32_t n=bkcz_plan_local_n(c.plan);c.local_n=uint8_t(n);
#pragma unroll
    for(uint32_t i=0;i<BKCZ_MAX_LOCAL;++i){
        if(i<n){uint32_t x=c.plan.local[i],sl=bkf_src_locator(x),ss=bkf_loc_owner(sl);BucketPhysicalBlock sb=bkf_high_main(ss,bkf_src_block(x));c.local_base[i]=bkf_ptr(ss,sb.off+Code(bkf_loc_rank(sl))*sb.cols);}
    }
    c.cross_base=nullptr;
    if(bkcz_plan_cross_depth(c.plan)){
        uint32_t x=bkcz_plan_cross_src(c.plan),sl=bkf_src_locator(x),ss=bkf_loc_owner(sl);BucketPhysicalBlock sb=bkf_high_main(ss,bkf_src_block(x));c.cross_base=bkf_ptr(ss,sb.off+Code(bkf_loc_rank(sl))*sb.cols);
    }
}

__device__ __forceinline__ BkczCrossAccum p10d8_sum_low_preimages_resolved(
    uint32_t dest_code,uint32_t depth,const Count* source_row
){
    BkczCrossAccum sum=0;int s=int(depth);uint32_t key=bkcz_ternary_key<LOW_LUT_K>(dest_code),weight=bkcz_pow3_const(LOW_LUT_K-1);
#pragma unroll
    for(int pos=LOW_LUT_K-1;pos>=0;--pos){
        uint32_t v=(dest_code>>(2*pos))&3u;
        if(v==uint32_t(R)){if(s==1)break;--s;}
        else if(v==uint32_t(::L)){
            if(s==1){uint32_t x=D_BKF_LOW_DIRECT[key-weight];if(x!=BKF_DIRECT_INVALID)sum=bkcz_cross_add(sum,source_row[bkf_loc_rank(x)]);}
            ++s;
        }
        if(pos)weight/=3u;
    }
    return sum;
}

__device__ __forceinline__ Count p10d8_highctx_plan_sum(
    const P10D8HighCtx& c,const BucketPhysicalBlock& db,uint32_t lr
){
#if GPU_DIRECT_PM_ACCUM
    uint64_t sum=0;
#else
    Count sum=0;
#endif
#pragma unroll
    for(uint32_t i=0;i<BKCZ_MAX_LOCAL;++i){
        if(i<c.local_n){Count v=c.local_base[i][lr];
#if GPU_DIRECT_PM_ACCUM
            sum+=uint64_t(v);
#else
            sum=gpu_direct_add(sum,v);
#endif
        }
    }
    uint32_t cross_depth=bkcz_plan_cross_depth(c.plan);
    if(cross_depth){uint32_t dc=D_BKF_LOW_CODES[D_BKF_LOW_CODE_OFF[size_t(D_BKF_FIXED_OWNER)*D_BKF_CODE_PITCH+db.hs]+lr];
#if GPU_DIRECT_PM_ACCUM
        sum+=p10d8_sum_low_preimages_resolved(dc,cross_depth,c.cross_base);
#else
        sum=gpu_direct_add(sum,p10d8_sum_low_preimages_resolved(dc,cross_depth,c.cross_base));
#endif
    }
#if GPU_DIRECT_PM_ACCUM
    return gpu_direct_pm_reduce_u64(sum);
#else
    return sum;
#endif
}
#else
__device__ __forceinline__ Count p10d8_highctx_plan_sum(
    const P10D8HighCtx& c,const BucketPhysicalBlock& db,uint32_t lr
){return bkcz_high_plan_sum(c.plan,db,lr);}
#endif

__global__ void bucket_high_orbit_closure_pattern10_depth8_highctx_kernel(int p){
    uint32_t bid=blockIdx.z;if(bid>=D_BKF_MAIN_NBLOCKS)return;uint32_t pi=uint32_t((TARGET_W-1)-p),oi=uint32_t(size_t(pi)*D_BKF_HIGH_PITCH+bid);__shared__ P10D8HighCtx c;
    if(threadIdx.x==0){uint32_t na=D_BKF_HIGH_NN_OFF[oi],nb=D_BKF_HIGH_NN_OFF[oi+1],ra=D_BKF_HIGH_NRNL_OFF[oi],rb=D_BKF_HIGH_NRNL_OFF[oi+1];c.n0=nb-na;c.n1=ra;c.total=c.n0+(rb-ra);}__syncthreads();
    for(uint32_t k=blockIdx.y;k<c.total;k+=gridDim.y){
        if(threadIdx.x==0){
            c.valid=0;bool nn=k<c.n0;uint32_t qi=nn?(D_BKF_HIGH_NN_OFF[oi]+k):(c.n1+k-c.n0);BucketOrbitOp op=nn?D_BKF_HIGH_NN[qi]:D_BKF_HIGH_NRNL[qi];uint8_t dep=nn?P10D8_HIGH_DEPTH_LOAD(D_P10D8_F_HIGH_NN,qi):P10D8_HIGH_DEPTH_LOAD(D_P10D8_F_HIGH_NRNL,qi);uint32_t sl=bkf_orbit_src(op),jl=bkf_orbit_partner(op),dl=bkf_orbit_drop(op);c.ss=bkf_loc_owner(sl);c.js=bkf_loc_owner(jl);c.ds=bkf_loc_owner(dl);c.xb=bkf_high_main(c.ss,bid);
            if(c.xb.valid&&c.xb.rows&&c.xb.cols){uint32_t jbid=bid;if(p==LOW_LUT_K+1){uint32_t center=nn?uint32_t(R):uint32_t(N);int he=int(c.xb.hs)+(center==uint32_t(R)?1:0);jbid=uint32_t(3*he+int(center));}c.jb=bkf_high_main(c.js,jbid);c.db=bkf_high_block(c.ds,uint32_t(c.xb.hs));c.sr=bkf_loc_rank(sl);c.jr=bkf_loc_rank(jl);c.dr=bkf_loc_rank(dl);c.kind=uint8_t(nn?CPU_ORBIT_NN:CPU_ORBIT_NR);c.plan=bkcpd8_forward_high(bkcp10_id(op),dep,dl,c.db,p);
#if P10D8_HIGHCTX_RESOLVE
                p10d8_highctx_resolve_plan(c);
#endif
                c.valid=1;}
        }
        __syncthreads();
        if(c.valid){for(uint32_t lr=uint32_t(blockIdx.x)*blockDim.x+threadIdx.x;lr<c.xb.cols;lr+=uint32_t(gridDim.x)*blockDim.x){Count*ip=bkf_ptr(c.ss,c.xb.off+Code(c.sr)*c.xb.cols+lr),*jp=bkf_ptr(c.js,c.jb.off+Code(c.jr)*c.jb.cols+lr),*dp=bkf_ptr(c.ds,c.db.off+Code(c.dr)*c.db.cols+lr);Count x=*ip,old=*dp,extra=p10d8_highctx_plan_sum(c,c.db,lr);if(c.kind==CPU_ORBIT_NN){*jp=gpu_direct_add(*jp,x);*ip=gpu_direct_add(x,old);*dp=extra;}else{Count y=*jp;*ip=gpu_direct_add(gpu_direct_add(x,y),old);*dp=gpu_direct_add(x,extra);}}}
        __syncthreads();
    }
}

__global__ void bucket_reverse_high_pattern10_depth8_highctx_kernel(int p){
    uint32_t bid=blockIdx.z;if(bid>=D_BKF_MAIN_NBLOCKS)return;uint32_t pi=uint32_t(p-(LOW_LUT_K+1)),oi=uint32_t(size_t(pi)*D_RS54_PITCH+bid);bool edge=p==TARGET_W-1;__shared__ P10D8HighCtx c;
    if(threadIdx.x==0){uint32_t na=D_RS54_HIGH_NN_OFF[oi],nb=D_RS54_HIGH_NN_OFF[oi+1],ra=D_RS54_HIGH_NR_OFF[oi],rb=D_RS54_HIGH_NR_OFF[oi+1],la=D_RS54_HIGH_NL_OFF[oi],lb=D_RS54_HIGH_NL_OFF[oi+1];c.n0=nb-na;c.n1=rb-ra;c.total=c.n0+c.n1+(lb-la);}__syncthreads();
    for(uint32_t k=blockIdx.y;k<c.total;k+=gridDim.y){
        if(threadIdx.x==0){
            c.valid=0;uint32_t qi=0,kind=0;BucketOrbitOp op;uint8_t dep;if(k<c.n0){kind=CPU_ORBIT_NN;qi=D_RS54_HIGH_NN_OFF[oi]+k;op=D_RS54_HIGH_NN[qi];dep=P10D8_HIGH_DEPTH_LOAD(D_P10D8_R_HIGH_NN,qi);}else if(k<c.n0+c.n1){kind=CPU_ORBIT_NR;qi=D_RS54_HIGH_NR_OFF[oi]+k-c.n0;op=D_RS54_HIGH_NR[qi];dep=P10D8_HIGH_DEPTH_LOAD(D_P10D8_R_HIGH_NR,qi);}else{kind=CPU_ORBIT_NL;qi=D_RS54_HIGH_NL_OFF[oi]+k-c.n0-c.n1;op=D_RS54_HIGH_NL[qi];dep=P10D8_HIGH_DEPTH_LOAD(D_P10D8_R_HIGH_NL,qi);}uint32_t sl=bkf_orbit_src(op),jl=bkf_orbit_partner(op),dl=bkf_orbit_drop(op);c.ss=bkf_loc_owner(sl);c.js=bkf_loc_owner(jl);c.ds=bkf_loc_owner(dl);c.xb=bkf_high_main(c.ss,bid);
            if(c.xb.valid&&c.xb.rows&&c.xb.cols){c.jb=bkf_high_main(c.js,bkcp10_reverse_high_jblock(bid,c.xb,p,kind));c.db=bkf_high_block(c.ds,uint32_t(c.xb.hs));c.sr=bkf_loc_rank(sl);c.jr=bkf_loc_rank(jl);c.dr=bkf_loc_rank(dl);c.kind=uint8_t(kind);c.plan=edge?bkcpd8_reverse_high(bkcp10_id(op),dep,sl,c.xb,p,true):bkcpd8_reverse_high(bkcp10_id(op),dep,dl,c.db,p,false);
#if P10D8_HIGHCTX_RESOLVE
                p10d8_highctx_resolve_plan(c);
#endif
                c.valid=1;}
        }
        __syncthreads();
        if(c.valid){for(uint32_t lr=uint32_t(blockIdx.x)*blockDim.x+threadIdx.x;lr<c.xb.cols;lr+=uint32_t(gridDim.x)*blockDim.x){Count*ip=bkf_ptr(c.ss,c.xb.off+Code(c.sr)*c.xb.cols+lr),*jp=bkf_ptr(c.js,c.jb.off+Code(c.jr)*c.jb.cols+lr),*dp=bkf_ptr(c.ds,c.db.off+Code(c.dr)*c.db.cols+lr);Count x=*ip,old=*dp,extra=p10d8_highctx_plan_sum(c,edge?c.xb:c.db,lr);if(c.kind==CPU_ORBIT_NN){*jp=gpu_direct_add(*jp,x);*ip=gpu_direct_add(gpu_direct_add(x,old),edge?extra:0);*dp=edge?0:extra;}else{Count y=*jp;*ip=gpu_direct_add(gpu_direct_add(x,y),old);if(edge){*jp=gpu_direct_add(x,y);*dp=0;}else *dp=gpu_direct_add(x,extra);}}}
        __syncthreads();
    }
}

static void bucket_launch_high_orbit_closure_pattern10_depth8_highctx(const StorageLayout&layout,int threads=256,int gx=16,int gy=8){dim3 block(threads),grid(gx,gy,unsigned(layout.main_blocks.size()));for(int p=TARGET_W-1;p>=LOW_LUT_K+1;--p){bucket_high_orbit_closure_pattern10_depth8_highctx_kernel<<<grid,block>>>(p);ck(cudaGetLastError(),"bucket high pattern10 depth8 highctx");}}
static void bucket_launch_reverse_high_pattern10_depth8_highctx(const StorageLayout&layout,int threads=256,int gx=16,int gy=8){dim3 block(threads),grid(gx,gy,unsigned(layout.main_blocks.size()));for(int p=LOW_LUT_K+1;p<TARGET_W;++p){bucket_reverse_high_pattern10_depth8_highctx_kernel<<<grid,block>>>(p);ck(cudaGetLastError(),"bucket reverse high pattern10 depth8 highctx");}}

#ifdef P10D8_HIGH_DEPTH_LOAD_LOCAL
#undef P10D8_HIGH_DEPTH_LOAD
#undef P10D8_HIGH_DEPTH_LOAD_LOCAL
#endif
