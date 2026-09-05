#pragma once

#include "ramstream32_bucket_orbit_closure_fused.cuh"
#include "ramstream32_bucket_orbit_closure_preflight.cuh"
#include "ramstream32_reverse_bucket_derive.hpp"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

// Production-oriented one-pass attachment encoding for W<=28.
//
// Destination block is implicit in (source block,p,kind), so only the
// destination-block-local closure-record ordinal is required.
// Reserve 18-bit 0x3ffff for NONE; real ordinals are 0..0x3fffe.
//
// Forward BucketOrbitOp has 10 spare bits (54..63):
//   low 10 bits of ordinal in the orbit word + high 8 bits in one byte/op.
//   Extra HBM = 1 B/op instead of 4 B/op.
//
// Reverse ReverseBucketOrbitOp has 6 redundant jblock bits plus 2 spare bits.
// After validating jblock derivability, bits 56..63 hold ordinal low 8 bits.
// Bits 8..15 live in one byte/op and bits 16..17 in a packed 2-bit stream.
//   Extra HBM = 1.25 B/op instead of 4 B/op.
static constexpr uint32_t BKOC18_NONE=0x3ffffu;
static constexpr uint64_t BKOC18_FORWARD_BASE_MASK=(1ull<<54)-1ull;
static constexpr uint64_t BKOC18_REVERSE_BASE_MASK=(1ull<<56)-1ull;

struct BucketForwardOrbitClosureAttach18Host{
    std::vector<uint8_t> low_nn,low_nr,low_nl,high_nn,high_nrnl;
    size_t bytes()const{return low_nn.size()+low_nr.size()+low_nl.size()+high_nn.size()+high_nrnl.size();}
};
struct BucketReverseOrbitClosureAttach18Host{
    std::vector<uint8_t> low_mid,high_mid,low_hi2,high_hi2;
    size_t bytes()const{return low_mid.size()+high_mid.size()+low_hi2.size()+high_hi2.size();}
};

static uint32_t bkoc18_local_code(uint32_t rid,const std::vector<uint32_t>&off,size_t oi,const char*what){
    if(rid==BKOC_NONE)return BKOC18_NONE;
    uint32_t a=off[oi],b=off[oi+1];
    if(rid<a||rid>=b){std::cerr<<"packed18 attachment outside destination block "<<what<<" rid="<<rid<<" range=["<<a<<','<<b<<")\n";std::exit(400);}
    uint32_t ord=rid-a;
    if(ord>=BKOC18_NONE){std::cerr<<"packed18 destination block overflow "<<what<<" records="<<(b-a)<<" ordinal="<<ord<<'\n';std::exit(401);}
    return ord;
}
static void bkoc18_set_hi2(std::vector<uint8_t>&v,size_t i,uint32_t x){size_t q=i>>2;unsigned sh=unsigned(i&3u)*2u;v[q]=uint8_t(v[q]|uint8_t((x&3u)<<sh));}
static uint32_t bkoc18_get_hi2_host(const std::vector<uint8_t>&v,size_t i){return (v[i>>2]>>(unsigned(i&3u)*2u))&3u;}

static BucketForwardOrbitClosureAttach18Host build_bucket_forward_orbit_closure_attach18(
    const StorageLayout&layout,BucketOrbitStreamsHost&bo,const BucketFusedHost&bf
){
    auto src=build_bucket_forward_orbit_closure_attach(layout,bo,bf);
    BucketForwardOrbitClosureAttach18Host out;
    out.low_nn.resize(bo.low_nn.size());out.low_nr.resize(bo.low_nr.size());out.low_nl.resize(bo.low_nl.size());out.high_nn.resize(bo.high_nn.size());out.high_nrnl.resize(bo.high_nrnl.size());
    size_t lp=size_t(bo.low_nblocks)+1,hp=size_t(bo.high_nblocks)+1;uint32_t maxrec=0;uint64_t attached=0;
    auto encode=[&](BucketOrbitOp&op,uint8_t&hi,uint32_t rid,const std::vector<uint32_t>&off,size_t doi,const char*what){uint32_t z=bkoc18_local_code(rid,off,doi,what);op=(op&BKOC18_FORWARD_BASE_MASK)|(uint64_t(z&0x3ffu)<<54);hi=uint8_t(z>>10);attached+=z!=BKOC18_NONE;};
    for(int p=LOW_LUT_K;p>=1;--p){uint32_t pi=uint32_t(LOW_LUT_K-p);for(uint32_t bid=0;bid<bo.low_nblocks;++bid){uint32_t dbid=p==1?bid:uint32_t(layout.main_blocks[bid].he);size_t doi=size_t(pi)*bf.low_pitch+dbid;maxrec=std::max(maxrec,bf.low_off[doi+1]-bf.low_off[doi]);auto cv=[&](auto&ops,const auto&s,std::vector<uint8_t>&hi,const auto&off){uint32_t a=off[size_t(pi)*lp+bid],b=off[size_t(pi)*lp+bid+1];for(uint32_t q=a;q<b;++q)encode(ops[q],hi[q],s[q],bf.low_off,doi,"forward-low");};cv(bo.low_nn,src.low_nn,out.low_nn,bo.low_nn_off);cv(bo.low_nr,src.low_nr,out.low_nr,bo.low_nr_off);cv(bo.low_nl,src.low_nl,out.low_nl,bo.low_nl_off);}}
    for(int p=TARGET_W-1;p>=LOW_LUT_K+1;--p){uint32_t pi=uint32_t((TARGET_W-1)-p);for(uint32_t bid=0;bid<bo.high_nblocks;++bid){uint32_t dbid=uint32_t(layout.main_blocks[bid].hs);size_t doi=size_t(pi)*bf.high_pitch+dbid;maxrec=std::max(maxrec,bf.high_off[doi+1]-bf.high_off[doi]);auto cv=[&](auto&ops,const auto&s,std::vector<uint8_t>&hi,const auto&off){uint32_t a=off[size_t(pi)*hp+bid],b=off[size_t(pi)*hp+bid+1];for(uint32_t q=a;q<b;++q)encode(ops[q],hi[q],s[q],bf.high_off,doi,"forward-high");};cv(bo.high_nn,src.high_nn,out.high_nn,bo.high_nn_off);cv(bo.high_nrnl,src.high_nrnl,out.high_nrnl,bo.high_nrnl_off);}}
    std::cerr<<"bucket_forward_orbit_closure_attach18 attached="<<attached<<" max_block_records="<<maxrec<<" sidecar_mib="<<double(out.bytes())/double(1<<20)<<" bytes_per_orbit=1\n";return out;
}

static BucketReverseOrbitClosureAttach18Host build_bucket_reverse_orbit_closure_attach18_checked(
    const StorageLayout&layout,const BucketOrbitStreamsHost&bo,const BucketFusedHost&bf,
    ReverseBucketAtomicHost&rb,const ReverseBucketFusedHost&rf
){
    auto src=build_bucket_reverse_orbit_closure_attach_checked(layout,bo,bf,rb,rf);
    // Must be done before overwriting bits 56..61.
    validate_reverse_bucket_partner_blocks(layout,rb);
    BucketReverseOrbitClosureAttach18Host out;out.low_mid.resize(rb.low_orbit.size());out.high_mid.resize(rb.high_orbit.size());out.low_hi2.assign((rb.low_orbit.size()+3)/4,0);out.high_hi2.assign((rb.high_orbit.size()+3)/4,0);
    size_t pitch=size_t(rb.nblocks)+1;uint32_t maxrec=0;uint64_t attached=0;
    auto encode=[&](ReverseBucketOrbitOp&op,uint8_t&mid,std::vector<uint8_t>&hi2,size_t q,uint32_t rid,const std::vector<uint32_t>&off,size_t doi,const char*what){uint32_t z=bkoc18_local_code(rid,off,doi,what);op=(op&BKOC18_REVERSE_BASE_MASK)|(uint64_t(z&0xffu)<<56);mid=uint8_t((z>>8)&0xffu);bkoc18_set_hi2(hi2,q,z>>16);attached+=z!=BKOC18_NONE;};
    for(int p=1;p<=LOW_LUT_K;++p){uint32_t pi=uint32_t(p-1);for(uint32_t bid=0;bid<rb.nblocks;++bid){uint32_t dbid=uint32_t(layout.main_blocks[bid].he);size_t doi=size_t(pi)*rf.low_pitch+dbid;maxrec=std::max(maxrec,rf.low_off[doi+1]-rf.low_off[doi]);uint32_t a=rb.low_orbit_off[size_t(pi)*pitch+bid],b=rb.low_orbit_off[size_t(pi)*pitch+bid+1];for(uint32_t q=a;q<b;++q)encode(rb.low_orbit[q],out.low_mid[q],out.low_hi2,q,src.low[q],rf.low_off,doi,"reverse-low");}}
    for(int p=LOW_LUT_K+1;p<TARGET_W;++p){uint32_t pi=uint32_t(p-(LOW_LUT_K+1));bool edge=p==TARGET_W-1;for(uint32_t bid=0;bid<rb.nblocks;++bid){uint32_t dbid=edge?bid:uint32_t(layout.main_blocks[bid].hs);size_t doi=size_t(pi)*rf.high_pitch+dbid;maxrec=std::max(maxrec,rf.high_off[doi+1]-rf.high_off[doi]);uint32_t a=rb.high_orbit_off[size_t(pi)*pitch+bid],b=rb.high_orbit_off[size_t(pi)*pitch+bid+1];for(uint32_t q=a;q<b;++q)encode(rb.high_orbit[q],out.high_mid[q],out.high_hi2,q,src.high[q],rf.high_off,doi,"reverse-high");}}
    std::cerr<<"bucket_reverse_orbit_closure_attach18 attached="<<attached<<" max_block_records="<<maxrec<<" sidecar_mib="<<double(out.bytes())/double(1<<20)<<" bytes_per_orbit="<<(rb.low_orbit.size()+rb.high_orbit.size()?double(out.bytes())/double(rb.low_orbit.size()+rb.high_orbit.size()):0.0)<<" jblock_bits_reused=6\n";return out;
}

__constant__ uint8_t *D_BKOC18_F_LOW_NN,*D_BKOC18_F_LOW_NR,*D_BKOC18_F_LOW_NL,*D_BKOC18_F_HIGH_NN,*D_BKOC18_F_HIGH_NRNL;
__constant__ uint8_t *D_BKOC18_R_LOW_MID,*D_BKOC18_R_HIGH_MID,*D_BKOC18_R_LOW_HI2,*D_BKOC18_R_HIGH_HI2;

struct BucketForwardOrbitClosureAttach18DeviceTables{
    uint8_t *low_nn=nullptr,*low_nr=nullptr,*low_nl=nullptr,*high_nn=nullptr,*high_nrnl=nullptr;
    static void cp(uint8_t*&d,const std::vector<uint8_t>&s,const char*w){if(s.empty())return;ck(cudaMalloc(&d,s.size()),w);ck(cudaMemcpy(d,s.data(),s.size(),cudaMemcpyHostToDevice),w);}
    void install(const BucketForwardOrbitClosureAttach18Host&h){cp(low_nn,h.low_nn,"bkoc18 f low nn");cp(low_nr,h.low_nr,"bkoc18 f low nr");cp(low_nl,h.low_nl,"bkoc18 f low nl");cp(high_nn,h.high_nn,"bkoc18 f high nn");cp(high_nrnl,h.high_nrnl,"bkoc18 f high nrnl");ck(cudaMemcpyToSymbol(D_BKOC18_F_LOW_NN,&low_nn,sizeof(low_nn)),"bkoc18 f low nn ptr");ck(cudaMemcpyToSymbol(D_BKOC18_F_LOW_NR,&low_nr,sizeof(low_nr)),"bkoc18 f low nr ptr");ck(cudaMemcpyToSymbol(D_BKOC18_F_LOW_NL,&low_nl,sizeof(low_nl)),"bkoc18 f low nl ptr");ck(cudaMemcpyToSymbol(D_BKOC18_F_HIGH_NN,&high_nn,sizeof(high_nn)),"bkoc18 f high nn ptr");ck(cudaMemcpyToSymbol(D_BKOC18_F_HIGH_NRNL,&high_nrnl,sizeof(high_nrnl)),"bkoc18 f high nrnl ptr");}
    void release(){cudaFree(low_nn);cudaFree(low_nr);cudaFree(low_nl);cudaFree(high_nn);cudaFree(high_nrnl);low_nn=low_nr=low_nl=high_nn=high_nrnl=nullptr;}
};
struct BucketReverseOrbitClosureAttach18DeviceTables{
    uint8_t *low_mid=nullptr,*high_mid=nullptr,*low_hi2=nullptr,*high_hi2=nullptr;
    static void cp(uint8_t*&d,const std::vector<uint8_t>&s,const char*w){if(s.empty())return;ck(cudaMalloc(&d,s.size()),w);ck(cudaMemcpy(d,s.data(),s.size(),cudaMemcpyHostToDevice),w);}
    void install(const BucketReverseOrbitClosureAttach18Host&h){cp(low_mid,h.low_mid,"bkoc18 r low mid");cp(high_mid,h.high_mid,"bkoc18 r high mid");cp(low_hi2,h.low_hi2,"bkoc18 r low hi2");cp(high_hi2,h.high_hi2,"bkoc18 r high hi2");ck(cudaMemcpyToSymbol(D_BKOC18_R_LOW_MID,&low_mid,sizeof(low_mid)),"bkoc18 r low mid ptr");ck(cudaMemcpyToSymbol(D_BKOC18_R_HIGH_MID,&high_mid,sizeof(high_mid)),"bkoc18 r high mid ptr");ck(cudaMemcpyToSymbol(D_BKOC18_R_LOW_HI2,&low_hi2,sizeof(low_hi2)),"bkoc18 r low hi2 ptr");ck(cudaMemcpyToSymbol(D_BKOC18_R_HIGH_HI2,&high_hi2,sizeof(high_hi2)),"bkoc18 r high hi2 ptr");}
    void release(){cudaFree(low_mid);cudaFree(high_mid);cudaFree(low_hi2);cudaFree(high_hi2);low_mid=high_mid=low_hi2=high_hi2=nullptr;}
};

__device__ __forceinline__ uint32_t bkoc18_forward_code(BucketOrbitOp op,uint8_t hi){return (uint32_t(hi)<<10)|uint32_t((op>>54)&0x3ffu);}
__device__ __forceinline__ uint32_t bkoc18_reverse_code(ReverseBucketOrbitOp op,uint8_t mid,const uint8_t*hi2,size_t q){uint32_t h=(hi2[q>>2]>>(unsigned(q&3u)*2u))&3u;return (h<<16)|(uint32_t(mid)<<8)|uint32_t((op>>56)&0xffu);}
__device__ __forceinline__ uint32_t bkoc18_rid(uint32_t z,const uint32_t*off,size_t oi){return z==BKOC18_NONE?BKOC_NONE:off[oi]+z;}
__device__ __forceinline__ uint32_t bkoc18_reverse_low_jblock(uint32_t bid,const BucketPhysicalBlock&xb,int p,uint32_t kind){if(p!=LOW_LUT_K)return bid;uint32_t center=kind==CPU_ORBIT_NN?uint32_t(::L):uint32_t(N);return 3u*uint32_t(xb.he)+center;}
__device__ __forceinline__ uint32_t bkoc18_reverse_high_jblock(uint32_t bid,const BucketPhysicalBlock&xb,int p,uint32_t kind){if(p!=LOW_LUT_K+1)return bid;uint32_t center=kind==CPU_ORBIT_NR?uint32_t(::L):uint32_t(R);int he=int(xb.hs)+(center==uint32_t(R)?1:-1);return uint32_t(3*he+int(center));}

__global__ void bucket_low_orbit_closure18_kernel(int p){uint32_t bid=blockIdx.z;if(bid>=D_BKF_MAIN_NBLOCKS)return;uint32_t pi=uint32_t(LOW_LUT_K-p);size_t oi=size_t(pi)*D_BKF_LOW_PITCH+bid;uint32_t na=D_BKF_LOW_NN_OFF[oi],nb=D_BKF_LOW_NN_OFF[oi+1],ra=D_BKF_LOW_NR_OFF[oi],rb=D_BKF_LOW_NR_OFF[oi+1],la=D_BKF_LOW_NL_OFF[oi],lb=D_BKF_LOW_NL_OFF[oi+1],n0=nb-na,n1=rb-ra,total=n0+n1+(lb-la);if(!total)return;for(uint32_t k=uint32_t(blockIdx.x)*blockDim.x+threadIdx.x;k<total;k+=uint32_t(gridDim.x)*blockDim.x){uint32_t kind;uint8_t hi;BucketOrbitOp op;if(k<n0){kind=CPU_ORBIT_NN;op=D_BKF_LOW_NN[na+k];hi=D_BKOC18_F_LOW_NN[na+k];}else if(k<n0+n1){kind=CPU_ORBIT_NR;op=D_BKF_LOW_NR[ra+k-n0];hi=D_BKOC18_F_LOW_NR[ra+k-n0];}else{kind=CPU_ORBIT_NL;op=D_BKF_LOW_NL[la+k-n0-n1];hi=D_BKOC18_F_LOW_NL[la+k-n0-n1];}uint32_t sl=bkf_orbit_src(op),jl=bkf_orbit_partner(op),dl=bkf_orbit_drop(op),ss=bkf_loc_owner(sl),js=bkf_loc_owner(jl),ds=bkf_loc_owner(dl);BucketPhysicalBlock xb=bkf_low_main(ss,bid);if(!xb.valid||!xb.rows||!xb.cols)continue;uint32_t jbid=bid;if(p==LOW_LUT_K){uint32_t center=kind==CPU_ORBIT_NR?uint32_t(R):uint32_t(::L);jbid=3u*uint32_t(xb.he)+center;}BucketPhysicalBlock jb=bkf_low_main(js,jbid),db=bkf_low_block(ds,uint32_t(xb.he));uint32_t dbid=p==1?bid:uint32_t(xb.he),rid=bkoc18_rid(bkoc18_forward_code(op,hi),D_BKF_LOW_OFF,size_t(pi)*D_BKF_LOW_FUSED_PITCH+dbid),sr=bkf_loc_rank(sl),jr=bkf_loc_rank(jl),dr=bkf_loc_rank(dl);for(uint32_t hr=blockIdx.y;hr<xb.rows;hr+=gridDim.y){Count*ip=bkf_ptr(ss,xb.off+Code(hr)*xb.cols+sr),*jp=bkf_ptr(js,jb.off+Code(hr)*jb.cols+jr),*dp=bkf_ptr(ds,db.off+Code(hr)*db.cols+dr);Count c=*ip,old=*dp,extra=bkoc_f_low_extra(rid,p==1?xb:db,hr);if(kind==CPU_ORBIT_NN){*jp=gpu_direct_add(*jp,c);*ip=gpu_direct_add(gpu_direct_add(c,old),p==1?extra:0);*dp=p==1?0:extra;}else{Count cc=*jp,all=gpu_direct_add(gpu_direct_add(c,cc),old);if(p==1){*ip=all;*jp=gpu_direct_add(c,cc);*dp=0;}else{*ip=all;*dp=gpu_direct_add(c,extra);}}}}}
__global__ void bucket_high_orbit_closure18_kernel(int p){uint32_t bid=blockIdx.z;if(bid>=D_BKF_MAIN_NBLOCKS)return;uint32_t pi=uint32_t((TARGET_W-1)-p);size_t oi=size_t(pi)*D_BKF_HIGH_PITCH+bid;uint32_t na=D_BKF_HIGH_NN_OFF[oi],nb=D_BKF_HIGH_NN_OFF[oi+1],ra=D_BKF_HIGH_NRNL_OFF[oi],rb=D_BKF_HIGH_NRNL_OFF[oi+1],n0=nb-na,total=n0+(rb-ra);if(!total)return;for(uint32_t k=blockIdx.y;k<total;k+=gridDim.y){bool nn=k<n0;uint32_t qi=nn?na+k:ra+k-n0;BucketOrbitOp op=nn?D_BKF_HIGH_NN[qi]:D_BKF_HIGH_NRNL[qi];uint8_t hi=nn?D_BKOC18_F_HIGH_NN[qi]:D_BKOC18_F_HIGH_NRNL[qi];uint32_t sl=bkf_orbit_src(op),jl=bkf_orbit_partner(op),dl=bkf_orbit_drop(op),ss=bkf_loc_owner(sl),js=bkf_loc_owner(jl),ds=bkf_loc_owner(dl);BucketPhysicalBlock xb=bkf_high_main(ss,bid);if(!xb.valid||!xb.rows||!xb.cols)continue;uint32_t jbid=bid;if(p==LOW_LUT_K+1){uint32_t center=nn?uint32_t(R):uint32_t(N);int he=int(xb.hs)+(center==uint32_t(R)?1:0);jbid=uint32_t(3*he+int(center));}BucketPhysicalBlock jb=bkf_high_main(js,jbid),db=bkf_high_block(ds,uint32_t(xb.hs));uint32_t dbid=uint32_t(xb.hs),rid=bkoc18_rid(bkoc18_forward_code(op,hi),D_BKF_HIGH_OFF,size_t(pi)*D_BKF_HIGH_FUSED_PITCH+dbid),sr=bkf_loc_rank(sl),jr=bkf_loc_rank(jl),dr=bkf_loc_rank(dl);for(uint32_t lr=uint32_t(blockIdx.x)*blockDim.x+threadIdx.x;lr<xb.cols;lr+=uint32_t(gridDim.x)*blockDim.x){Count*ip=bkf_ptr(ss,xb.off+Code(sr)*xb.cols+lr),*jp=bkf_ptr(js,jb.off+Code(jr)*jb.cols+lr),*dp=bkf_ptr(ds,db.off+Code(dr)*db.cols+lr);Count c=*ip,old=*dp,extra=bkoc_f_high_extra(rid,db,lr);if(nn){*jp=gpu_direct_add(*jp,c);*ip=gpu_direct_add(c,old);*dp=extra;}else{Count cc=*jp;*ip=gpu_direct_add(gpu_direct_add(c,cc),old);*dp=gpu_direct_add(c,extra);}}}}
__global__ void bucket_reverse_low_orbit_closure18_kernel(int p){uint32_t bid=blockIdx.z;if(bid>=D_BKF_MAIN_NBLOCKS)return;uint32_t pi=uint32_t(p-1);size_t oi=size_t(pi)*D_RB_PITCH+bid;uint32_t a=D_RB_LOW_ORBIT_OFF[oi],b=D_RB_LOW_ORBIT_OFF[oi+1];for(uint32_t q=a+uint32_t(blockIdx.x)*blockDim.x+threadIdx.x;q<b;q+=uint32_t(gridDim.x)*blockDim.x){uint64_t w=D_RB_LOW_ORBIT[q];uint32_t z=bkoc18_reverse_code(w,D_BKOC18_R_LOW_MID[q],D_BKOC18_R_LOW_HI2,q),sl=rb_orbit_src(w),jl=rb_orbit_partner(w),dl=rb_orbit_drop(w),ss=bkf_loc_owner(sl),js=bkf_loc_owner(jl),ds=bkf_loc_owner(dl),kind=rb_orbit_kind(w);BucketPhysicalBlock xb=bkf_low_main(ss,bid),jb=bkf_low_main(js,bkoc18_reverse_low_jblock(bid,xb,p,kind)),db=bkf_low_block(ds,uint32_t(xb.he));uint32_t dbid=uint32_t(xb.he),rid=bkoc18_rid(z,D_RBF_LOW_OFF,size_t(pi)*D_RBF_LOW_PITCH+dbid);for(uint32_t hr=blockIdx.y;hr<xb.rows;hr+=gridDim.y){Count*ip=bkf_ptr(ss,xb.off+Code(hr)*xb.cols+bkf_loc_rank(sl)),*jp=bkf_ptr(js,jb.off+Code(hr)*jb.cols+bkf_loc_rank(jl)),*dp=bkf_ptr(ds,db.off+Code(hr)*db.cols+bkf_loc_rank(dl));Count c=*ip,old=*dp,extra=bkoc_r_low_extra(rid,db,hr);if(kind==CPU_ORBIT_NN){*jp=gpu_direct_add(*jp,c);*ip=gpu_direct_add(c,old);*dp=extra;}else{Count cc=*jp;*ip=gpu_direct_add(gpu_direct_add(c,cc),old);*dp=gpu_direct_add(c,extra);}}}}
__global__ void bucket_reverse_high_orbit_closure18_kernel(int p){uint32_t bid=blockIdx.z;if(bid>=D_BKF_MAIN_NBLOCKS)return;uint32_t pi=uint32_t(p-(LOW_LUT_K+1));size_t oi=size_t(pi)*D_RB_PITCH+bid;uint32_t a=D_RB_HIGH_ORBIT_OFF[oi],b=D_RB_HIGH_ORBIT_OFF[oi+1];bool edge=p==TARGET_W-1;for(uint32_t q=a+blockIdx.y;q<b;q+=gridDim.y){uint64_t w=D_RB_HIGH_ORBIT[q];uint32_t z=bkoc18_reverse_code(w,D_BKOC18_R_HIGH_MID[q],D_BKOC18_R_HIGH_HI2,q),sl=rb_orbit_src(w),jl=rb_orbit_partner(w),dl=rb_orbit_drop(w),ss=bkf_loc_owner(sl),js=bkf_loc_owner(jl),ds=bkf_loc_owner(dl),kind=rb_orbit_kind(w);BucketPhysicalBlock xb=bkf_high_main(ss,bid),jb=bkf_high_main(js,bkoc18_reverse_high_jblock(bid,xb,p,kind)),db=bkf_high_block(ds,uint32_t(xb.hs));uint32_t dbid=edge?bid:uint32_t(xb.hs),rid=bkoc18_rid(z,D_RBF_HIGH_OFF,size_t(pi)*D_RBF_HIGH_PITCH+dbid);for(uint32_t lr=uint32_t(blockIdx.x)*blockDim.x+threadIdx.x;lr<xb.cols;lr+=uint32_t(gridDim.x)*blockDim.x){Count*ip=bkf_ptr(ss,xb.off+Code(bkf_loc_rank(sl))*xb.cols+lr),*jp=bkf_ptr(js,jb.off+Code(bkf_loc_rank(jl))*jb.cols+lr),*dp=bkf_ptr(ds,db.off+Code(bkf_loc_rank(dl))*db.cols+lr);Count c=*ip,old=*dp,extra=bkoc_r_high_extra(rid,edge?xb:db,lr);if(kind==CPU_ORBIT_NN){*jp=gpu_direct_add(*jp,c);*ip=gpu_direct_add(gpu_direct_add(c,old),edge?extra:0);*dp=edge?0:extra;}else{Count cc=*jp;*ip=gpu_direct_add(gpu_direct_add(c,cc),old);if(edge){*jp=gpu_direct_add(c,cc);*dp=0;}else *dp=gpu_direct_add(c,extra);}}}}

static void bucket_launch_low_orbit_closure18(const StorageLayout&layout,int threads=256,int gx=16,int gy=8){dim3 block(threads),grid(gx,gy,unsigned(layout.main_blocks.size()));for(int p=LOW_LUT_K;p>=1;--p){bucket_low_orbit_closure18_kernel<<<grid,block>>>(p);ck(cudaGetLastError(),"bucket low orbit closure18");}}
static void bucket_launch_high_orbit_closure18(const StorageLayout&layout,int threads=256,int gx=16,int gy=8){dim3 block(threads),grid(gx,gy,unsigned(layout.main_blocks.size()));for(int p=TARGET_W-1;p>=LOW_LUT_K+1;--p){bucket_high_orbit_closure18_kernel<<<grid,block>>>(p);ck(cudaGetLastError(),"bucket high orbit closure18");}}
static void bucket_launch_reverse_low_orbit_closure18(const StorageLayout&layout,int threads=256,int gx=16,int gy=8){dim3 block(threads),grid(gx,gy,unsigned(layout.main_blocks.size()));for(int p=1;p<=LOW_LUT_K;++p){bucket_reverse_low_orbit_closure18_kernel<<<grid,block>>>(p);ck(cudaGetLastError(),"bucket reverse low orbit closure18");}}
static void bucket_launch_reverse_high_orbit_closure18(const StorageLayout&layout,int threads=256,int gx=16,int gy=8){dim3 block(threads),grid(gx,gy,unsigned(layout.main_blocks.size()));for(int p=LOW_LUT_K+1;p<TARGET_W;++p){bucket_reverse_high_orbit_closure18_kernel<<<grid,block>>>(p);ck(cudaGetLastError(),"bucket reverse high orbit closure18");}}
