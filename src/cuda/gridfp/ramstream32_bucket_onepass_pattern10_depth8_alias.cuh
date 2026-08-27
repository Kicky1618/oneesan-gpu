#pragma once

#include "ramstream32_bucket_closure_pattern10_depth8.hpp"
#include "ramstream32_bucket_onepass_zero_alias.cuh"

struct BucketForwardPattern10Depth8Host {
    std::vector<uint8_t> low_nn,low_nr,low_nl,high_nn,high_nrnl;
    size_t bytes()const{return low_nn.size()+low_nr.size()+low_nl.size()+high_nn.size()+high_nrnl.size();}
};
struct BucketReversePattern10Depth8Host {
    ReverseSplit54Host split;
    std::vector<uint8_t> low_nn,low_nr,low_nl,high_nn,high_nr,high_nl;
    size_t bytes()const{return split.bytes()+low_nn.size()+low_nr.size()+low_nl.size()+high_nn.size()+high_nr.size()+high_nl.size();}
};

static BucketForwardPattern10Depth8Host build_bucket_forward_pattern10_depth8_zero(
    const StorageLayout&layout,BucketOrbitStreamsHost&bo,BucketFusedHost&bf
){
    build_bucket_forward_pattern10(layout,bo,bf);BucketForwardPattern10Depth8Host out;
    out.low_nn.resize(bo.low_nn.size());out.low_nr.resize(bo.low_nr.size());out.low_nl.resize(bo.low_nl.size());out.high_nn.resize(bo.high_nn.size());out.high_nrnl.resize(bo.high_nrnl.size());
    uint64_t nz=0;uint8_t md=0;auto put=[&](auto&v,uint32_t q,uint8_t d){v[q]=d;nz+=d!=0;md=std::max(md,d);};size_t lp=size_t(bo.low_nblocks)+1,hp=size_t(bo.high_nblocks)+1;
    for(int p=LOW_LUT_K;p>=1;--p){uint32_t pi=uint32_t(LOW_LUT_K-p);for(uint32_t bid=0;bid<bo.low_nblocks;++bid){const auto&xb=layout.main_blocks[bid];if(!xb.valid)continue;const auto&db=p==1?xb:layout.block_blocks[xb.he];auto scan=[&](const auto&ops,const auto&off,auto&dst,bool active){for(uint32_t q=off[size_t(pi)*lp+bid];q<off[size_t(pi)*lp+bid+1];++q){uint8_t dep=0;if(active){uint32_t loc=p==1?bkf_orbit_src(ops[q]):bkf_orbit_drop(ops[q]),dc=bkcp10_low_code_host(bf,loc,db.hs);MateID d=p==1?(MateID(dc)|(MateID(db.c)<<(2*LOW_LUT_K))):minsert(MateID(dc),p,N);dep=bkcp10_depth8_low_host(d,p,"forward-low");}put(dst,q,dep);}};scan(bo.low_nn,bo.low_nn_off,out.low_nn,true);scan(bo.low_nr,bo.low_nr_off,out.low_nr,p!=1);scan(bo.low_nl,bo.low_nl_off,out.low_nl,p!=1);}}
    for(int p=TARGET_W-1;p>=LOW_LUT_K+1;--p){uint32_t pi=uint32_t((TARGET_W-1)-p);int rel=p-LOW_LUT_K;for(uint32_t bid=0;bid<bo.high_nblocks;++bid){const auto&xb=layout.main_blocks[bid];if(!xb.valid)continue;const auto&db=layout.block_blocks[xb.hs];auto scan=[&](const auto&ops,const auto&off,auto&dst){for(uint32_t q=off[size_t(pi)*hp+bid];q<off[size_t(pi)*hp+bid+1];++q){uint32_t loc=bkf_orbit_drop(ops[q]),dc=bkcp10_high_code_host(bf,loc,db.he);MateID d=minsert(MateID(dc),rel,N);put(dst,q,bkcp10_depth8_high_host(d,rel,"forward-high"));}};scan(bo.high_nn,bo.high_nn_off,out.high_nn);scan(bo.high_nrnl,bo.high_nrnl_off,out.high_nrnl);}}
    std::cerr<<"bucket_forward_pattern10_depth8 bytes="<<out.bytes()<<" nonzero_cross="<<nz<<" max_depth="<<unsigned(md)<<" bytes_per_orbit=1\n";bucket_zero_release_forward_closure(bf);return out;
}

static BucketReversePattern10Depth8Host build_bucket_reverse_pattern10_depth8_zero_checked(
    const StorageLayout&layout,const BucketOrbitStreamsHost&,BucketFusedHost&bf,
    ReverseBucketAtomicHost&rb,ReverseBucketFusedHost&rf
){
    BucketReversePattern10Depth8Host out;out.split=build_reverse_split54(layout,rb,true);build_reverse_split54_pattern10(layout,bf,out.split);auto&rs=out.split;
    out.low_nn.resize(rs.low.nn.size());out.low_nr.resize(rs.low.nr.size());out.low_nl.resize(rs.low.nl.size());out.high_nn.resize(rs.high.nn.size());out.high_nr.resize(rs.high.nr.size());out.high_nl.resize(rs.high.nl.size());
    uint64_t nz=0;uint8_t md=0;auto put=[&](auto&v,uint32_t q,uint8_t d){v[q]=d;nz+=d!=0;md=std::max(md,d);};size_t rp=size_t(rs.nblocks)+1;
    for(int p=1;p<=LOW_LUT_K;++p){uint32_t pi=uint32_t(p-1);for(uint32_t bid=0;bid<rs.nblocks;++bid){const auto&xb=layout.main_blocks[bid];if(!xb.valid)continue;const auto&db=layout.block_blocks[xb.he];auto scan=[&](const auto&ops,const auto&off,auto&dst){for(uint32_t q=off[size_t(pi)*rp+bid];q<off[size_t(pi)*rp+bid+1];++q){uint32_t loc=bkf_orbit_drop(ops[q]),dc=bkcp10_low_code_host(bf,loc,db.hs);MateID d=blocked_exclude_reverse(MateID(dc),LOW_LUT_K+1,p);put(dst,q,bkcp10_depth8_low_host(d,p,"reverse-low"));}};scan(rs.low.nn,rs.low.nn_off,out.low_nn);scan(rs.low.nr,rs.low.nr_off,out.low_nr);scan(rs.low.nl,rs.low.nl_off,out.low_nl);}}
    for(int p=LOW_LUT_K+1;p<TARGET_W;++p){uint32_t pi=uint32_t(p-(LOW_LUT_K+1));int rel=p-LOW_LUT_K;bool edge=p==TARGET_W-1;for(uint32_t bid=0;bid<rs.nblocks;++bid){const auto&xb=layout.main_blocks[bid];if(!xb.valid)continue;const auto&db=edge?xb:layout.block_blocks[xb.hs];auto scan=[&](const auto&ops,const auto&off,auto&dst,bool nn){for(uint32_t q=off[size_t(pi)*rp+bid];q<off[size_t(pi)*rp+bid+1];++q){uint8_t dep=0;if(!edge||nn){uint32_t loc=edge?bkf_orbit_src(ops[q]):bkf_orbit_drop(ops[q]),dc=bkcp10_high_code_host(bf,loc,db.he);MateID d=edge?(MateID(db.c)|(MateID(dc)<<2)):blocked_exclude_reverse(MateID(dc),HIGH_LUT_K+1,rel);dep=bkcp10_depth8_high_host(d,rel,"reverse-high");}put(dst,q,dep);}};scan(rs.high.nn,rs.high.nn_off,out.high_nn,true);scan(rs.high.nr,rs.high.nr_off,out.high_nr,false);scan(rs.high.nl,rs.high.nl_off,out.high_nl,false);}}
    std::cerr<<"bucket_reverse_pattern10_depth8 bytes="<<out.bytes()<<" depth_bytes="<<(out.bytes()-out.split.bytes())<<" nonzero_cross="<<nz<<" max_depth="<<unsigned(md)<<" bytes_per_orbit=1\n";bucket_zero_release_reverse_closure(rf);return out;
}

__constant__ uint8_t *D_P10D8_F_LOW_NN,*D_P10D8_F_LOW_NR,*D_P10D8_F_LOW_NL,*D_P10D8_F_HIGH_NN,*D_P10D8_F_HIGH_NRNL;
__constant__ uint8_t *D_P10D8_R_LOW_NN,*D_P10D8_R_LOW_NR,*D_P10D8_R_LOW_NL,*D_P10D8_R_HIGH_NN,*D_P10D8_R_HIGH_NR,*D_P10D8_R_HIGH_NL;

struct BucketForwardPattern10Depth8DeviceTables {
    uint8_t *low_nn=nullptr,*low_nr=nullptr,*low_nl=nullptr,*high_nn=nullptr,*high_nrnl=nullptr;
    static void cp(uint8_t*&d,const std::vector<uint8_t>&s,const char*w){if(s.empty())return;ck(cudaMalloc(&d,s.size()),w);ck(cudaMemcpy(d,s.data(),s.size(),cudaMemcpyHostToDevice),w);}
    void install(const BucketForwardPattern10Depth8Host&h){cp(low_nn,h.low_nn,"p10d8 f low nn");cp(low_nr,h.low_nr,"p10d8 f low nr");cp(low_nl,h.low_nl,"p10d8 f low nl");cp(high_nn,h.high_nn,"p10d8 f high nn");cp(high_nrnl,h.high_nrnl,"p10d8 f high nrnl");ck(cudaMemcpyToSymbol(D_P10D8_F_LOW_NN,&low_nn,sizeof(low_nn)),"p10d8 f low nn ptr");ck(cudaMemcpyToSymbol(D_P10D8_F_LOW_NR,&low_nr,sizeof(low_nr)),"p10d8 f low nr ptr");ck(cudaMemcpyToSymbol(D_P10D8_F_LOW_NL,&low_nl,sizeof(low_nl)),"p10d8 f low nl ptr");ck(cudaMemcpyToSymbol(D_P10D8_F_HIGH_NN,&high_nn,sizeof(high_nn)),"p10d8 f high nn ptr");ck(cudaMemcpyToSymbol(D_P10D8_F_HIGH_NRNL,&high_nrnl,sizeof(high_nrnl)),"p10d8 f high nrnl ptr");}
    void release(){cudaFree(low_nn);cudaFree(low_nr);cudaFree(low_nl);cudaFree(high_nn);cudaFree(high_nrnl);low_nn=low_nr=low_nl=high_nn=high_nrnl=nullptr;}
};
struct BucketReversePattern10Depth8DeviceTables {
    ReverseSplit54DeviceTables split;uint8_t *low_nn=nullptr,*low_nr=nullptr,*low_nl=nullptr,*high_nn=nullptr,*high_nr=nullptr,*high_nl=nullptr;
    static void cp(uint8_t*&d,const std::vector<uint8_t>&s,const char*w){if(s.empty())return;ck(cudaMalloc(&d,s.size()),w);ck(cudaMemcpy(d,s.data(),s.size(),cudaMemcpyHostToDevice),w);}
    void install(const BucketReversePattern10Depth8Host&h){split.install(h.split);cp(low_nn,h.low_nn,"p10d8 r low nn");cp(low_nr,h.low_nr,"p10d8 r low nr");cp(low_nl,h.low_nl,"p10d8 r low nl");cp(high_nn,h.high_nn,"p10d8 r high nn");cp(high_nr,h.high_nr,"p10d8 r high nr");cp(high_nl,h.high_nl,"p10d8 r high nl");ck(cudaMemcpyToSymbol(D_P10D8_R_LOW_NN,&low_nn,sizeof(low_nn)),"p10d8 r low nn ptr");ck(cudaMemcpyToSymbol(D_P10D8_R_LOW_NR,&low_nr,sizeof(low_nr)),"p10d8 r low nr ptr");ck(cudaMemcpyToSymbol(D_P10D8_R_LOW_NL,&low_nl,sizeof(low_nl)),"p10d8 r low nl ptr");ck(cudaMemcpyToSymbol(D_P10D8_R_HIGH_NN,&high_nn,sizeof(high_nn)),"p10d8 r high nn ptr");ck(cudaMemcpyToSymbol(D_P10D8_R_HIGH_NR,&high_nr,sizeof(high_nr)),"p10d8 r high nr ptr");ck(cudaMemcpyToSymbol(D_P10D8_R_HIGH_NL,&high_nl,sizeof(high_nl)),"p10d8 r high nl ptr");}
    void release(){cudaFree(low_nn);cudaFree(low_nr);cudaFree(low_nl);cudaFree(high_nn);cudaFree(high_nr);cudaFree(high_nl);low_nn=low_nr=low_nl=high_nn=high_nr=high_nl=nullptr;split.release();}
};
