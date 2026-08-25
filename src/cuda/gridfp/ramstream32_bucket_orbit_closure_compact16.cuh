#pragma once

#include "ramstream32_bucket_orbit_closure_fused.cuh"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

// Compact one-pass attachment metadata.
//
// The original one-pass prototype stores one global uint32_t closure-record ID
// per orbit operation.  The destination block is already determined by
// (source block, p, orbit kind), and each fused destination stream is contiguous
// per destination block.  Store only the destination-block-local ordinal.
// 0xffff is NONE, so a destination block may contain at most 65535 records.
// This cuts attachment HBM from 4 B/op to 2 B/op without changing the fused
// destination records or the authoritative layout.
static constexpr uint16_t BKOC16_NONE = 0xffffu;

struct BucketForwardOrbitClosureAttach16Host {
    std::vector<uint16_t> low_nn,low_nr,low_nl,high_nn,high_nrnl;
    size_t bytes() const {
        return (low_nn.size()+low_nr.size()+low_nl.size()+high_nn.size()+high_nrnl.size())*sizeof(uint16_t);
    }
};
struct BucketReverseOrbitClosureAttach16Host {
    std::vector<uint16_t> low,high;
    size_t bytes() const { return (low.size()+high.size())*sizeof(uint16_t); }
};

static uint16_t bkoc16_local_ordinal(
    uint32_t rid,const std::vector<uint32_t>&off,size_t oi,const char*what
){
    if(rid==BKOC_NONE)return BKOC16_NONE;
    uint32_t a=off[oi],b=off[oi+1];
    if(rid<a||rid>=b){
        std::cerr<<"compact16 attachment outside destination block "<<what
                 <<" rid="<<rid<<" range=["<<a<<','<<b<<")\n";
        std::exit(390);
    }
    uint32_t ord=rid-a;
    if(ord>=uint32_t(BKOC16_NONE)){
        std::cerr<<"compact16 destination block overflow "<<what
                 <<" records="<<(b-a)<<" ordinal="<<ord<<'\n';
        std::exit(391);
    }
    return uint16_t(ord);
}

static BucketForwardOrbitClosureAttach16Host build_bucket_forward_orbit_closure_attach16(
    const StorageLayout&layout,const BucketOrbitStreamsHost&bo,const BucketFusedHost&bf,
    const BucketForwardOrbitClosureAttachHost&src
){
    BucketForwardOrbitClosureAttach16Host out;
    out.low_nn.assign(src.low_nn.size(),BKOC16_NONE);
    out.low_nr.assign(src.low_nr.size(),BKOC16_NONE);
    out.low_nl.assign(src.low_nl.size(),BKOC16_NONE);
    out.high_nn.assign(src.high_nn.size(),BKOC16_NONE);
    out.high_nrnl.assign(src.high_nrnl.size(),BKOC16_NONE);
    size_t lp=size_t(bo.low_nblocks)+1,hp=size_t(bo.high_nblocks)+1;
    uint32_t max_low=0,max_high=0;uint64_t attached=0;

    for(int p=LOW_LUT_K;p>=1;--p){
        uint32_t pi=uint32_t(LOW_LUT_K-p);
        for(uint32_t bid=0;bid<bo.low_nblocks;++bid){
            uint32_t dbid=p==1?bid:uint32_t(layout.main_blocks[bid].he);
            size_t doi=size_t(pi)*bf.low_pitch+dbid;
            max_low=std::max(max_low,bf.low_off[doi+1]-bf.low_off[doi]);
            auto cv=[&](const std::vector<uint32_t>&s,std::vector<uint16_t>&d,const std::vector<uint32_t>&off){
                uint32_t a=off[size_t(pi)*lp+bid],b=off[size_t(pi)*lp+bid+1];
                for(uint32_t q=a;q<b;++q){d[q]=bkoc16_local_ordinal(s[q],bf.low_off,doi,"forward-low");attached+=d[q]!=BKOC16_NONE;}
            };
            cv(src.low_nn,out.low_nn,bo.low_nn_off);
            cv(src.low_nr,out.low_nr,bo.low_nr_off);
            cv(src.low_nl,out.low_nl,bo.low_nl_off);
        }
    }
    for(int p=TARGET_W-1;p>=LOW_LUT_K+1;--p){
        uint32_t pi=uint32_t((TARGET_W-1)-p);
        for(uint32_t bid=0;bid<bo.high_nblocks;++bid){
            uint32_t dbid=uint32_t(layout.main_blocks[bid].hs);
            size_t doi=size_t(pi)*bf.high_pitch+dbid;
            max_high=std::max(max_high,bf.high_off[doi+1]-bf.high_off[doi]);
            auto cv=[&](const std::vector<uint32_t>&s,std::vector<uint16_t>&d,const std::vector<uint32_t>&off){
                uint32_t a=off[size_t(pi)*hp+bid],b=off[size_t(pi)*hp+bid+1];
                for(uint32_t q=a;q<b;++q){d[q]=bkoc16_local_ordinal(s[q],bf.high_off,doi,"forward-high");attached+=d[q]!=BKOC16_NONE;}
            };
            cv(src.high_nn,out.high_nn,bo.high_nn_off);
            cv(src.high_nrnl,out.high_nrnl,bo.high_nrnl_off);
        }
    }
    std::cerr<<"bucket_forward_orbit_closure_attach16 attached="<<attached
             <<" max_low_block_records="<<max_low<<" max_high_block_records="<<max_high
             <<" mib="<<double(out.bytes())/double(1<<20)<<'\n';
    return out;
}

static BucketReverseOrbitClosureAttach16Host build_bucket_reverse_orbit_closure_attach16(
    const StorageLayout&layout,const ReverseBucketAtomicHost&rb,const ReverseBucketFusedHost&rf,
    const BucketReverseOrbitClosureAttachHost&src
){
    BucketReverseOrbitClosureAttach16Host out;
    out.low.assign(src.low.size(),BKOC16_NONE);out.high.assign(src.high.size(),BKOC16_NONE);
    size_t pitch=size_t(rb.nblocks)+1;uint32_t max_low=0,max_high=0;uint64_t attached=0;
    for(int p=1;p<=LOW_LUT_K;++p){
        uint32_t pi=uint32_t(p-1);
        for(uint32_t bid=0;bid<rb.nblocks;++bid){
            uint32_t dbid=uint32_t(layout.main_blocks[bid].he);size_t doi=size_t(pi)*rf.low_pitch+dbid;
            max_low=std::max(max_low,rf.low_off[doi+1]-rf.low_off[doi]);
            uint32_t a=rb.low_orbit_off[size_t(pi)*pitch+bid],b=rb.low_orbit_off[size_t(pi)*pitch+bid+1];
            for(uint32_t q=a;q<b;++q){out.low[q]=bkoc16_local_ordinal(src.low[q],rf.low_off,doi,"reverse-low");attached+=out.low[q]!=BKOC16_NONE;}
        }
    }
    for(int p=LOW_LUT_K+1;p<TARGET_W;++p){
        uint32_t pi=uint32_t(p-(LOW_LUT_K+1));bool edge=p==TARGET_W-1;
        for(uint32_t bid=0;bid<rb.nblocks;++bid){
            uint32_t dbid=edge?bid:uint32_t(layout.main_blocks[bid].hs);size_t doi=size_t(pi)*rf.high_pitch+dbid;
            max_high=std::max(max_high,rf.high_off[doi+1]-rf.high_off[doi]);
            uint32_t a=rb.high_orbit_off[size_t(pi)*pitch+bid],b=rb.high_orbit_off[size_t(pi)*pitch+bid+1];
            for(uint32_t q=a;q<b;++q){out.high[q]=bkoc16_local_ordinal(src.high[q],rf.high_off,doi,"reverse-high");attached+=out.high[q]!=BKOC16_NONE;}
        }
    }
    std::cerr<<"bucket_reverse_orbit_closure_attach16 attached="<<attached
             <<" max_low_block_records="<<max_low<<" max_high_block_records="<<max_high
             <<" mib="<<double(out.bytes())/double(1<<20)<<'\n';
    return out;
}

__constant__ uint16_t *D_BKOC16_F_LOW_NN,*D_BKOC16_F_LOW_NR,*D_BKOC16_F_LOW_NL,*D_BKOC16_F_HIGH_NN,*D_BKOC16_F_HIGH_NRNL;
__constant__ uint16_t *D_BKOC16_R_LOW,*D_BKOC16_R_HIGH;

struct BucketForwardOrbitClosureAttach16DeviceTables {
    uint16_t *low_nn=nullptr,*low_nr=nullptr,*low_nl=nullptr,*high_nn=nullptr,*high_nrnl=nullptr;
    static void cp(uint16_t*&d,const std::vector<uint16_t>&s,const char*w){if(s.empty())return;ck(cudaMalloc(&d,s.size()*sizeof(uint16_t)),w);ck(cudaMemcpy(d,s.data(),s.size()*sizeof(uint16_t),cudaMemcpyHostToDevice),w);}
    void install(const BucketForwardOrbitClosureAttach16Host&h){cp(low_nn,h.low_nn,"bkoc16 f low nn");cp(low_nr,h.low_nr,"bkoc16 f low nr");cp(low_nl,h.low_nl,"bkoc16 f low nl");cp(high_nn,h.high_nn,"bkoc16 f high nn");cp(high_nrnl,h.high_nrnl,"bkoc16 f high nrnl");ck(cudaMemcpyToSymbol(D_BKOC16_F_LOW_NN,&low_nn,sizeof(low_nn)),"bkoc16 f low nn ptr");ck(cudaMemcpyToSymbol(D_BKOC16_F_LOW_NR,&low_nr,sizeof(low_nr)),"bkoc16 f low nr ptr");ck(cudaMemcpyToSymbol(D_BKOC16_F_LOW_NL,&low_nl,sizeof(low_nl)),"bkoc16 f low nl ptr");ck(cudaMemcpyToSymbol(D_BKOC16_F_HIGH_NN,&high_nn,sizeof(high_nn)),"bkoc16 f high nn ptr");ck(cudaMemcpyToSymbol(D_BKOC16_F_HIGH_NRNL,&high_nrnl,sizeof(high_nrnl)),"bkoc16 f high nrnl ptr");}
    void release(){cudaFree(low_nn);cudaFree(low_nr);cudaFree(low_nl);cudaFree(high_nn);cudaFree(high_nrnl);low_nn=low_nr=low_nl=high_nn=high_nrnl=nullptr;}
};
struct BucketReverseOrbitClosureAttach16DeviceTables {
    uint16_t *low=nullptr,*high=nullptr;
    static void cp(uint16_t*&d,const std::vector<uint16_t>&s,const char*w){if(s.empty())return;ck(cudaMalloc(&d,s.size()*sizeof(uint16_t)),w);ck(cudaMemcpy(d,s.data(),s.size()*sizeof(uint16_t),cudaMemcpyHostToDevice),w);}
    void install(const BucketReverseOrbitClosureAttach16Host&h){cp(low,h.low,"bkoc16 r low");cp(high,h.high,"bkoc16 r high");ck(cudaMemcpyToSymbol(D_BKOC16_R_LOW,&low,sizeof(low)),"bkoc16 r low ptr");ck(cudaMemcpyToSymbol(D_BKOC16_R_HIGH,&high,sizeof(high)),"bkoc16 r high ptr");}
    void release(){cudaFree(low);cudaFree(high);low=high=nullptr;}
};

__device__ __forceinline__ uint32_t bkoc16_rid(uint16_t ord,const uint32_t*off,size_t oi){return ord==BKOC16_NONE?BKOC_NONE:off[oi]+uint32_t(ord);}
__device__ __forceinline__ uint32_t bkoc16_reverse_low_jblock(uint32_t bid,const BucketPhysicalBlock&xb,int p,uint32_t kind){if(p!=LOW_LUT_K)return bid;uint32_t center=kind==CPU_ORBIT_NN?uint32_t(::L):uint32_t(N);return 3u*uint32_t(xb.he)+center;}
__device__ __forceinline__ uint32_t bkoc16_reverse_high_jblock(uint32_t bid,const BucketPhysicalBlock&xb,int p,uint32_t kind){if(p!=LOW_LUT_K+1)return bid;uint32_t center=kind==CPU_ORBIT_NR?uint32_t(::L):uint32_t(R);int he=int(xb.hs)+(center==uint32_t(R)?1:-1);return uint32_t(3*he+int(center));}

__global__ void bucket_low_orbit_closure16_kernel(int p){
    uint32_t bid=blockIdx.z;if(bid>=D_BKF_MAIN_NBLOCKS)return;uint32_t pi=uint32_t(LOW_LUT_K-p);size_t oi=size_t(pi)*D_BKF_LOW_PITCH+bid;
    uint32_t na=D_BKF_LOW_NN_OFF[oi],nb=D_BKF_LOW_NN_OFF[oi+1],ra=D_BKF_LOW_NR_OFF[oi],rb=D_BKF_LOW_NR_OFF[oi+1],la=D_BKF_LOW_NL_OFF[oi],lb=D_BKF_LOW_NL_OFF[oi+1];uint32_t n0=nb-na,n1=rb-ra,total=n0+n1+(lb-la);if(!total)return;
    for(uint32_t k=uint32_t(blockIdx.x)*blockDim.x+threadIdx.x;k<total;k+=uint32_t(gridDim.x)*blockDim.x){uint32_t kind;uint16_t ord;BucketOrbitOp op;if(k<n0){kind=CPU_ORBIT_NN;op=D_BKF_LOW_NN[na+k];ord=D_BKOC16_F_LOW_NN[na+k];}else if(k<n0+n1){kind=CPU_ORBIT_NR;op=D_BKF_LOW_NR[ra+k-n0];ord=D_BKOC16_F_LOW_NR[ra+k-n0];}else{kind=CPU_ORBIT_NL;op=D_BKF_LOW_NL[la+k-n0-n1];ord=D_BKOC16_F_LOW_NL[la+k-n0-n1];}
        uint32_t sl=bkf_orbit_src(op),jl=bkf_orbit_partner(op),dl=bkf_orbit_drop(op),ss=bkf_loc_owner(sl),js=bkf_loc_owner(jl),ds=bkf_loc_owner(dl);BucketPhysicalBlock xb=bkf_low_main(ss,bid);if(!xb.valid||!xb.rows||!xb.cols)continue;uint32_t jbid=bid;if(p==LOW_LUT_K){uint32_t center=kind==CPU_ORBIT_NR?uint32_t(R):uint32_t(::L);jbid=3u*uint32_t(xb.he)+center;}BucketPhysicalBlock jb=bkf_low_main(js,jbid),db=bkf_low_block(ds,uint32_t(xb.he));uint32_t dbid=p==1?bid:uint32_t(xb.he),rid=bkoc16_rid(ord,D_BKF_LOW_OFF,size_t(pi)*D_BKF_LOW_FUSED_PITCH+dbid),sr=bkf_loc_rank(sl),jr=bkf_loc_rank(jl),dr=bkf_loc_rank(dl);
        for(uint32_t hr=blockIdx.y;hr<xb.rows;hr+=gridDim.y){Count*ip=bkf_ptr(ss,xb.off+Code(hr)*xb.cols+sr),*jp=bkf_ptr(js,jb.off+Code(hr)*jb.cols+jr),*dp=bkf_ptr(ds,db.off+Code(hr)*db.cols+dr);Count c=*ip,old=*dp,extra=bkoc_f_low_extra(rid,p==1?xb:db,hr);if(kind==CPU_ORBIT_NN){*jp=gpu_direct_add(*jp,c);*ip=gpu_direct_add(gpu_direct_add(c,old),p==1?extra:0);*dp=p==1?0:extra;}else{Count cc=*jp,all=gpu_direct_add(gpu_direct_add(c,cc),old);if(p==1){*ip=all;*jp=gpu_direct_add(c,cc);*dp=0;}else{*ip=all;*dp=gpu_direct_add(c,extra);}}}
    }
}
__global__ void bucket_high_orbit_closure16_kernel(int p){
    uint32_t bid=blockIdx.z;if(bid>=D_BKF_MAIN_NBLOCKS)return;uint32_t pi=uint32_t((TARGET_W-1)-p);size_t oi=size_t(pi)*D_BKF_HIGH_PITCH+bid;uint32_t na=D_BKF_HIGH_NN_OFF[oi],nb=D_BKF_HIGH_NN_OFF[oi+1],ra=D_BKF_HIGH_NRNL_OFF[oi],rb=D_BKF_HIGH_NRNL_OFF[oi+1],n0=nb-na,total=n0+(rb-ra);if(!total)return;
    for(uint32_t k=blockIdx.y;k<total;k+=gridDim.y){bool nn=k<n0;uint32_t qi=nn?na+k:ra+k-n0;BucketOrbitOp op=nn?D_BKF_HIGH_NN[qi]:D_BKF_HIGH_NRNL[qi];uint16_t ord=nn?D_BKOC16_F_HIGH_NN[qi]:D_BKOC16_F_HIGH_NRNL[qi];uint32_t sl=bkf_orbit_src(op),jl=bkf_orbit_partner(op),dl=bkf_orbit_drop(op),ss=bkf_loc_owner(sl),js=bkf_loc_owner(jl),ds=bkf_loc_owner(dl);BucketPhysicalBlock xb=bkf_high_main(ss,bid);if(!xb.valid||!xb.rows||!xb.cols)continue;uint32_t jbid=bid;if(p==LOW_LUT_K+1){uint32_t center=nn?uint32_t(R):uint32_t(N);int he=int(xb.hs)+(center==uint32_t(R)?1:0);jbid=uint32_t(3*he+int(center));}BucketPhysicalBlock jb=bkf_high_main(js,jbid),db=bkf_high_block(ds,uint32_t(xb.hs));uint32_t dbid=uint32_t(xb.hs),rid=bkoc16_rid(ord,D_BKF_HIGH_OFF,size_t(pi)*D_BKF_HIGH_FUSED_PITCH+dbid),sr=bkf_loc_rank(sl),jr=bkf_loc_rank(jl),dr=bkf_loc_rank(dl);
        for(uint32_t lr=uint32_t(blockIdx.x)*blockDim.x+threadIdx.x;lr<xb.cols;lr+=uint32_t(gridDim.x)*blockDim.x){Count*ip=bkf_ptr(ss,xb.off+Code(sr)*xb.cols+lr),*jp=bkf_ptr(js,jb.off+Code(jr)*jb.cols+lr),*dp=bkf_ptr(ds,db.off+Code(dr)*db.cols+lr);Count c=*ip,old=*dp,extra=bkoc_f_high_extra(rid,db,lr);if(nn){*jp=gpu_direct_add(*jp,c);*ip=gpu_direct_add(c,old);*dp=extra;}else{Count cc=*jp;*ip=gpu_direct_add(gpu_direct_add(c,cc),old);*dp=gpu_direct_add(c,extra);}}
    }
}
__global__ void bucket_reverse_low_orbit_closure16_kernel(int p){
    uint32_t bid=blockIdx.z;if(bid>=D_BKF_MAIN_NBLOCKS)return;uint32_t pi=uint32_t(p-1);size_t oi=size_t(pi)*D_RB_PITCH+bid;uint32_t a=D_RB_LOW_ORBIT_OFF[oi],b=D_RB_LOW_ORBIT_OFF[oi+1];
    for(uint32_t q=a+uint32_t(blockIdx.x)*blockDim.x+threadIdx.x;q<b;q+=uint32_t(gridDim.x)*blockDim.x){uint64_t w=D_RB_LOW_ORBIT[q];uint16_t ord=D_BKOC16_R_LOW[q];uint32_t sl=rb_orbit_src(w),jl=rb_orbit_partner(w),dl=rb_orbit_drop(w),ss=bkf_loc_owner(sl),js=bkf_loc_owner(jl),ds=bkf_loc_owner(dl),kind=rb_orbit_kind(w);BucketPhysicalBlock xb=bkf_low_main(ss,bid),jb=bkf_low_main(js,bkoc16_reverse_low_jblock(bid,xb,p,kind)),db=bkf_low_block(ds,uint32_t(xb.he));uint32_t dbid=uint32_t(xb.he),rid=bkoc16_rid(ord,D_RBF_LOW_OFF,size_t(pi)*D_RBF_LOW_PITCH+dbid);
        for(uint32_t hr=blockIdx.y;hr<xb.rows;hr+=gridDim.y){Count*ip=bkf_ptr(ss,xb.off+Code(hr)*xb.cols+bkf_loc_rank(sl)),*jp=bkf_ptr(js,jb.off+Code(hr)*jb.cols+bkf_loc_rank(jl)),*dp=bkf_ptr(ds,db.off+Code(hr)*db.cols+bkf_loc_rank(dl));Count c=*ip,old=*dp,extra=bkoc_r_low_extra(rid,db,hr);if(kind==CPU_ORBIT_NN){*jp=gpu_direct_add(*jp,c);*ip=gpu_direct_add(c,old);*dp=extra;}else{Count cc=*jp;*ip=gpu_direct_add(gpu_direct_add(c,cc),old);*dp=gpu_direct_add(c,extra);}}
    }
}
__global__ void bucket_reverse_high_orbit_closure16_kernel(int p){
    uint32_t bid=blockIdx.z;if(bid>=D_BKF_MAIN_NBLOCKS)return;uint32_t pi=uint32_t(p-(LOW_LUT_K+1));size_t oi=size_t(pi)*D_RB_PITCH+bid;uint32_t a=D_RB_HIGH_ORBIT_OFF[oi],b=D_RB_HIGH_ORBIT_OFF[oi+1];bool edge=p==TARGET_W-1;
    for(uint32_t q=a+blockIdx.y;q<b;q+=gridDim.y){uint64_t w=D_RB_HIGH_ORBIT[q];uint16_t ord=D_BKOC16_R_HIGH[q];uint32_t sl=rb_orbit_src(w),jl=rb_orbit_partner(w),dl=rb_orbit_drop(w),ss=bkf_loc_owner(sl),js=bkf_loc_owner(jl),ds=bkf_loc_owner(dl),kind=rb_orbit_kind(w);BucketPhysicalBlock xb=bkf_high_main(ss,bid),jb=bkf_high_main(js,bkoc16_reverse_high_jblock(bid,xb,p,kind)),db=bkf_high_block(ds,uint32_t(xb.hs));uint32_t dbid=edge?bid:uint32_t(xb.hs),rid=bkoc16_rid(ord,D_RBF_HIGH_OFF,size_t(pi)*D_RBF_HIGH_PITCH+dbid);
        for(uint32_t lr=uint32_t(blockIdx.x)*blockDim.x+threadIdx.x;lr<xb.cols;lr+=uint32_t(gridDim.x)*blockDim.x){Count*ip=bkf_ptr(ss,xb.off+Code(bkf_loc_rank(sl))*xb.cols+lr),*jp=bkf_ptr(js,jb.off+Code(bkf_loc_rank(jl))*jb.cols+lr),*dp=bkf_ptr(ds,db.off+Code(bkf_loc_rank(dl))*db.cols+lr);Count c=*ip,old=*dp,extra=bkoc_r_high_extra(rid,edge?xb:db,lr);if(kind==CPU_ORBIT_NN){*jp=gpu_direct_add(*jp,c);*ip=gpu_direct_add(gpu_direct_add(c,old),edge?extra:0);*dp=edge?0:extra;}else{Count cc=*jp;*ip=gpu_direct_add(gpu_direct_add(c,cc),old);if(edge){*jp=gpu_direct_add(c,cc);*dp=0;}else *dp=gpu_direct_add(c,extra);}}
    }
}

static void bucket_launch_low_orbit_closure16(const StorageLayout&layout,int threads=256,int gx=16,int gy=8){dim3 block(threads),grid(gx,gy,unsigned(layout.main_blocks.size()));for(int p=LOW_LUT_K;p>=1;--p){bucket_low_orbit_closure16_kernel<<<grid,block>>>(p);ck(cudaGetLastError(),"bucket low orbit closure16");}}
static void bucket_launch_high_orbit_closure16(const StorageLayout&layout,int threads=256,int gx=16,int gy=8){dim3 block(threads),grid(gx,gy,unsigned(layout.main_blocks.size()));for(int p=TARGET_W-1;p>=LOW_LUT_K+1;--p){bucket_high_orbit_closure16_kernel<<<grid,block>>>(p);ck(cudaGetLastError(),"bucket high orbit closure16");}}
static void bucket_launch_reverse_low_orbit_closure16(const StorageLayout&layout,int threads=256,int gx=16,int gy=8){dim3 block(threads),grid(gx,gy,unsigned(layout.main_blocks.size()));for(int p=1;p<=LOW_LUT_K;++p){bucket_reverse_low_orbit_closure16_kernel<<<grid,block>>>(p);ck(cudaGetLastError(),"bucket reverse low orbit closure16");}}
static void bucket_launch_reverse_high_orbit_closure16(const StorageLayout&layout,int threads=256,int gx=16,int gy=8){dim3 block(threads),grid(gx,gy,unsigned(layout.main_blocks.size()));for(int p=LOW_LUT_K+1;p<TARGET_W;++p){bucket_reverse_high_orbit_closure16_kernel<<<grid,block>>>(p);ck(cudaGetLastError(),"bucket reverse high orbit closure16");}}
