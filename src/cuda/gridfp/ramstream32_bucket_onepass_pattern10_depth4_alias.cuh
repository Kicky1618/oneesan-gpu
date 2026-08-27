#pragma once

#include "ramstream32_bucket_onepass_pattern10_depth8_alias.cuh"

struct BucketDepth4Host {
    std::vector<uint8_t> packed;
    size_t count=0;
    size_t bytes()const{return packed.size();}
};
static BucketDepth4Host bkcpd4_pack(std::vector<uint8_t>&&src,const char*what){
    BucketDepth4Host out;out.count=src.size();out.packed.assign((src.size()+1)/2,0);
    for(size_t i=0;i<src.size();++i){uint8_t d=src[i];if(d>15){std::cerr<<"pattern10 depth4 overflow "<<what<<" q="<<i<<" depth="<<unsigned(d)<<'\n';std::exit(582);}out.packed[i>>1]|=uint8_t(d<<((i&1u)*4));}
    return out;
}

struct BucketForwardPattern10Depth4Host {
    BucketDepth4Host low_nn,low_nr,low_nl,high_nn,high_nrnl;
    size_t bytes()const{return low_nn.bytes()+low_nr.bytes()+low_nl.bytes()+high_nn.bytes()+high_nrnl.bytes();}
};
struct BucketReversePattern10Depth4Host {
    ReverseSplit54Host split;
    BucketDepth4Host low_nn,low_nr,low_nl,high_nn,high_nr,high_nl;
    size_t bytes()const{return split.bytes()+low_nn.bytes()+low_nr.bytes()+low_nl.bytes()+high_nn.bytes()+high_nr.bytes()+high_nl.bytes();}
};

static BucketForwardPattern10Depth4Host build_bucket_forward_pattern10_depth4_zero(
    const StorageLayout&layout,BucketOrbitStreamsHost&bo,BucketFusedHost&bf
){
    auto h=build_bucket_forward_pattern10_depth8_zero(layout,bo,bf);BucketForwardPattern10Depth4Host out;
    out.low_nn=bkcpd4_pack(std::move(h.low_nn),"forward-low-nn");out.low_nr=bkcpd4_pack(std::move(h.low_nr),"forward-low-nr");out.low_nl=bkcpd4_pack(std::move(h.low_nl),"forward-low-nl");out.high_nn=bkcpd4_pack(std::move(h.high_nn),"forward-high-nn");out.high_nrnl=bkcpd4_pack(std::move(h.high_nrnl),"forward-high-nrnl");
    std::cerr<<"bucket_forward_pattern10_depth4 bytes="<<out.bytes()<<" bytes_per_orbit=0.5_padded\n";return out;
}
static BucketReversePattern10Depth4Host build_bucket_reverse_pattern10_depth4_zero_checked(
    const StorageLayout&layout,const BucketOrbitStreamsHost&bo,BucketFusedHost&bf,
    ReverseBucketAtomicHost&rb,ReverseBucketFusedHost&rf
){
    auto h=build_bucket_reverse_pattern10_depth8_zero_checked(layout,bo,bf,rb,rf);BucketReversePattern10Depth4Host out;out.split=std::move(h.split);
    out.low_nn=bkcpd4_pack(std::move(h.low_nn),"reverse-low-nn");out.low_nr=bkcpd4_pack(std::move(h.low_nr),"reverse-low-nr");out.low_nl=bkcpd4_pack(std::move(h.low_nl),"reverse-low-nl");out.high_nn=bkcpd4_pack(std::move(h.high_nn),"reverse-high-nn");out.high_nr=bkcpd4_pack(std::move(h.high_nr),"reverse-high-nr");out.high_nl=bkcpd4_pack(std::move(h.high_nl),"reverse-high-nl");
    std::cerr<<"bucket_reverse_pattern10_depth4 bytes="<<out.bytes()<<" depth_bytes="<<(out.bytes()-out.split.bytes())<<" bytes_per_orbit=0.5_padded\n";return out;
}

struct BucketForwardPattern10Depth4DeviceTables {
    uint8_t *low_nn=nullptr,*low_nr=nullptr,*low_nl=nullptr,*high_nn=nullptr,*high_nrnl=nullptr;
    static void cp(uint8_t*&d,const BucketDepth4Host&s,const char*w){if(s.packed.empty())return;ck(cudaMalloc(&d,s.packed.size()),w);ck(cudaMemcpy(d,s.packed.data(),s.packed.size(),cudaMemcpyHostToDevice),w);}
    void install(const BucketForwardPattern10Depth4Host&h){cp(low_nn,h.low_nn,"p10d4 f low nn");cp(low_nr,h.low_nr,"p10d4 f low nr");cp(low_nl,h.low_nl,"p10d4 f low nl");cp(high_nn,h.high_nn,"p10d4 f high nn");cp(high_nrnl,h.high_nrnl,"p10d4 f high nrnl");ck(cudaMemcpyToSymbol(D_P10D8_F_LOW_NN,&low_nn,sizeof(low_nn)),"p10d4 f low nn ptr");ck(cudaMemcpyToSymbol(D_P10D8_F_LOW_NR,&low_nr,sizeof(low_nr)),"p10d4 f low nr ptr");ck(cudaMemcpyToSymbol(D_P10D8_F_LOW_NL,&low_nl,sizeof(low_nl)),"p10d4 f low nl ptr");ck(cudaMemcpyToSymbol(D_P10D8_F_HIGH_NN,&high_nn,sizeof(high_nn)),"p10d4 f high nn ptr");ck(cudaMemcpyToSymbol(D_P10D8_F_HIGH_NRNL,&high_nrnl,sizeof(high_nrnl)),"p10d4 f high nrnl ptr");}
    void release(){cudaFree(low_nn);cudaFree(low_nr);cudaFree(low_nl);cudaFree(high_nn);cudaFree(high_nrnl);low_nn=low_nr=low_nl=high_nn=high_nrnl=nullptr;}
};
struct BucketReversePattern10Depth4DeviceTables {
    ReverseSplit54DeviceTables split;uint8_t *low_nn=nullptr,*low_nr=nullptr,*low_nl=nullptr,*high_nn=nullptr,*high_nr=nullptr,*high_nl=nullptr;
    static void cp(uint8_t*&d,const BucketDepth4Host&s,const char*w){if(s.packed.empty())return;ck(cudaMalloc(&d,s.packed.size()),w);ck(cudaMemcpy(d,s.packed.data(),s.packed.size(),cudaMemcpyHostToDevice),w);}
    void install(const BucketReversePattern10Depth4Host&h){split.install(h.split);cp(low_nn,h.low_nn,"p10d4 r low nn");cp(low_nr,h.low_nr,"p10d4 r low nr");cp(low_nl,h.low_nl,"p10d4 r low nl");cp(high_nn,h.high_nn,"p10d4 r high nn");cp(high_nr,h.high_nr,"p10d4 r high nr");cp(high_nl,h.high_nl,"p10d4 r high nl");ck(cudaMemcpyToSymbol(D_P10D8_R_LOW_NN,&low_nn,sizeof(low_nn)),"p10d4 r low nn ptr");ck(cudaMemcpyToSymbol(D_P10D8_R_LOW_NR,&low_nr,sizeof(low_nr)),"p10d4 r low nr ptr");ck(cudaMemcpyToSymbol(D_P10D8_R_LOW_NL,&low_nl,sizeof(low_nl)),"p10d4 r low nl ptr");ck(cudaMemcpyToSymbol(D_P10D8_R_HIGH_NN,&high_nn,sizeof(high_nn)),"p10d4 r high nn ptr");ck(cudaMemcpyToSymbol(D_P10D8_R_HIGH_NR,&high_nr,sizeof(high_nr)),"p10d4 r high nr ptr");ck(cudaMemcpyToSymbol(D_P10D8_R_HIGH_NL,&high_nl,sizeof(high_nl)),"p10d4 r high nl ptr");}
    void release(){cudaFree(low_nn);cudaFree(low_nr);cudaFree(low_nl);cudaFree(high_nn);cudaFree(high_nr);cudaFree(high_nl);low_nn=low_nr=low_nl=high_nn=high_nr=high_nl=nullptr;split.release();}
};
