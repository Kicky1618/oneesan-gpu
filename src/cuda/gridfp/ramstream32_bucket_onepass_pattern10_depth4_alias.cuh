#pragma once

#include "ramstream32_bucket_onepass_pattern10_depth8_alias.cuh"

struct BucketDepth4Host {
    std::vector<uint8_t> packed;
    size_t count=0;
    void resize(size_t n){count=n;packed.assign((n+1)/2,0);}
    void set(size_t q,uint8_t d,const char*what){if(q>=count||d>15){std::cerr<<"pattern10 depth4 invalid "<<what<<" q="<<q<<" count="<<count<<" depth="<<unsigned(d)<<'\n';std::exit(582);}uint32_t sh=uint32_t(q&1u)*4u;packed[q>>1]=uint8_t((packed[q>>1]&uint8_t(~(0xfu<<sh)))|uint8_t(d<<sh));}
    uint8_t get(size_t q)const{return q<count?uint8_t((packed[q>>1]>>((q&1u)*4u))&0xfu):0;}
    size_t bytes()const{return packed.size();}
};
static BucketDepth4Host bkcpd4_pack(std::vector<uint8_t>&&src,const char*what){BucketDepth4Host out;out.resize(src.size());for(size_t i=0;i<src.size();++i)out.set(i,src[i],what);return out;}

struct BucketForwardPattern10Depth4Host {
    BucketDepth4Host low_nn,low_nr,low_nl,high_nn,high_nrnl;
    size_t bytes()const{return low_nn.bytes()+low_nr.bytes()+low_nl.bytes()+high_nn.bytes()+high_nrnl.bytes();}
    size_t ops()const{return low_nn.count+low_nr.count+low_nl.count+high_nn.count+high_nrnl.count;}
};
struct BucketReversePattern10Depth4Host {
    ReverseSplit54Host split;
    BucketDepth4Host low_nn,low_nr,low_nl,high_nn,high_nr,high_nl;
    size_t bytes()const{return split.bytes()+low_nn.bytes()+low_nr.bytes()+low_nl.bytes()+high_nn.bytes()+high_nr.bytes()+high_nl.bytes();}
    size_t depth_bytes()const{return low_nn.bytes()+low_nr.bytes()+low_nl.bytes()+high_nn.bytes()+high_nr.bytes()+high_nl.bytes();}
    size_t ops()const{return low_nn.count+low_nr.count+low_nl.count+high_nn.count+high_nr.count+high_nl.count;}
};

static BucketForwardPattern10Depth4Host build_bucket_forward_pattern10_depth4_zero(
    const StorageLayout&layout,BucketOrbitStreamsHost&bo,BucketFusedHost&bf
){
    build_bucket_forward_pattern10(layout,bo,bf);BucketForwardPattern10Depth4Host out;
    out.low_nn.resize(bo.low_nn.size());out.low_nr.resize(bo.low_nr.size());out.low_nl.resize(bo.low_nl.size());out.high_nn.resize(bo.high_nn.size());out.high_nrnl.resize(bo.high_nrnl.size());
    uint64_t nz=0;uint8_t md=0;auto put=[&](BucketDepth4Host&v,uint32_t q,uint8_t d,const char*w){v.set(q,d,w);nz+=d!=0;md=std::max(md,d);};size_t lp=size_t(bo.low_nblocks)+1,hp=size_t(bo.high_nblocks)+1;
    for(int p=LOW_LUT_K;p>=1;--p){uint32_t pi=uint32_t(LOW_LUT_K-p);for(uint32_t bid=0;bid<bo.low_nblocks;++bid){const auto&xb=layout.main_blocks[bid];if(!xb.valid)continue;const auto&db=p==1?xb:layout.block_blocks[xb.he];auto scan=[&](const auto&ops,const auto&off,BucketDepth4Host&dst,bool active,const char*w){for(uint32_t q=off[size_t(pi)*lp+bid];q<off[size_t(pi)*lp+bid+1];++q){uint8_t dep=0;if(active){uint32_t loc=p==1?bkf_orbit_src(ops[q]):bkf_orbit_drop(ops[q]),dc=bkcp10_low_code_host(bf,loc,db.hs);MateID d=p==1?(MateID(dc)|(MateID(db.c)<<(2*LOW_LUT_K))):minsert(MateID(dc),p,N);dep=bkcp10_depth8_low_host(d,p,"forward-low");}put(dst,q,dep,w);}};scan(bo.low_nn,bo.low_nn_off,out.low_nn,true,"forward-low-nn");scan(bo.low_nr,bo.low_nr_off,out.low_nr,p!=1,"forward-low-nr");scan(bo.low_nl,bo.low_nl_off,out.low_nl,p!=1,"forward-low-nl");}}
    for(int p=TARGET_W-1;p>=LOW_LUT_K+1;--p){uint32_t pi=uint32_t((TARGET_W-1)-p);int rel=p-LOW_LUT_K;for(uint32_t bid=0;bid<bo.high_nblocks;++bid){const auto&xb=layout.main_blocks[bid];if(!xb.valid)continue;const auto&db=layout.block_blocks[xb.hs];auto scan=[&](const auto&ops,const auto&off,BucketDepth4Host&dst,const char*w){for(uint32_t q=off[size_t(pi)*hp+bid];q<off[size_t(pi)*hp+bid+1];++q){uint32_t loc=bkf_orbit_drop(ops[q]),dc=bkcp10_high_code_host(bf,loc,db.he);MateID d=minsert(MateID(dc),rel,N);put(dst,q,bkcp10_depth8_high_host(d,rel,"forward-high"),w);}};scan(bo.high_nn,bo.high_nn_off,out.high_nn,"forward-high-nn");scan(bo.high_nrnl,bo.high_nrnl_off,out.high_nrnl,"forward-high-nrnl");}}
    std::cerr<<"bucket_forward_pattern10_depth4 ops="<<out.ops()<<" bytes="<<out.bytes()<<" nonzero_cross="<<nz<<" max_depth="<<unsigned(md)<<" bytes_per_orbit=0.5_padded direct_build=1\n";bucket_zero_release_forward_closure(bf);return out;
}

static BucketReversePattern10Depth4Host build_bucket_reverse_pattern10_depth4_zero_checked(
    const StorageLayout&layout,const BucketOrbitStreamsHost&,BucketFusedHost&bf,
    ReverseBucketAtomicHost&rb,ReverseBucketFusedHost&rf
){
    BucketReversePattern10Depth4Host out;out.split=build_reverse_split54(layout,rb,true);build_reverse_split54_pattern10(layout,bf,out.split);auto&rs=out.split;
    out.low_nn.resize(rs.low.nn.size());out.low_nr.resize(rs.low.nr.size());out.low_nl.resize(rs.low.nl.size());out.high_nn.resize(rs.high.nn.size());out.high_nr.resize(rs.high.nr.size());out.high_nl.resize(rs.high.nl.size());
    uint64_t nz=0;uint8_t md=0;auto put=[&](BucketDepth4Host&v,uint32_t q,uint8_t d,const char*w){v.set(q,d,w);nz+=d!=0;md=std::max(md,d);};size_t rp=size_t(rs.nblocks)+1;
    for(int p=1;p<=LOW_LUT_K;++p){uint32_t pi=uint32_t(p-1);for(uint32_t bid=0;bid<rs.nblocks;++bid){const auto&xb=layout.main_blocks[bid];if(!xb.valid)continue;const auto&db=layout.block_blocks[xb.he];auto scan=[&](const auto&ops,const auto&off,BucketDepth4Host&dst,const char*w){for(uint32_t q=off[size_t(pi)*rp+bid];q<off[size_t(pi)*rp+bid+1];++q){uint32_t loc=bkf_orbit_drop(ops[q]),dc=bkcp10_low_code_host(bf,loc,db.hs);MateID d=blocked_exclude_reverse(MateID(dc),LOW_LUT_K+1,p);put(dst,q,bkcp10_depth8_low_host(d,p,"reverse-low"),w);}};scan(rs.low.nn,rs.low.nn_off,out.low_nn,"reverse-low-nn");scan(rs.low.nr,rs.low.nr_off,out.low_nr,"reverse-low-nr");scan(rs.low.nl,rs.low.nl_off,out.low_nl,"reverse-low-nl");}}
    for(int p=LOW_LUT_K+1;p<TARGET_W;++p){uint32_t pi=uint32_t(p-(LOW_LUT_K+1));int rel=p-LOW_LUT_K;bool edge=p==TARGET_W-1;for(uint32_t bid=0;bid<rs.nblocks;++bid){const auto&xb=layout.main_blocks[bid];if(!xb.valid)continue;const auto&db=edge?xb:layout.block_blocks[xb.hs];auto scan=[&](const auto&ops,const auto&off,BucketDepth4Host&dst,bool nn,const char*w){for(uint32_t q=off[size_t(pi)*rp+bid];q<off[size_t(pi)*rp+bid+1];++q){uint8_t dep=0;if(!edge||nn){uint32_t loc=edge?bkf_orbit_src(ops[q]):bkf_orbit_drop(ops[q]),dc=bkcp10_high_code_host(bf,loc,db.he);MateID d=edge?(MateID(db.c)|(MateID(dc)<<2)):blocked_exclude_reverse(MateID(dc),HIGH_LUT_K+1,rel);dep=bkcp10_depth8_high_host(d,rel,"reverse-high");}put(dst,q,dep,w);}};scan(rs.high.nn,rs.high.nn_off,out.high_nn,true,"reverse-high-nn");scan(rs.high.nr,rs.high.nr_off,out.high_nr,false,"reverse-high-nr");scan(rs.high.nl,rs.high.nl_off,out.high_nl,false,"reverse-high-nl");}}
    std::cerr<<"bucket_reverse_pattern10_depth4 ops="<<out.ops()<<" bytes="<<out.bytes()<<" depth_bytes="<<out.depth_bytes()<<" nonzero_cross="<<nz<<" max_depth="<<unsigned(md)<<" bytes_per_orbit=0.5_padded direct_build=1\n";bucket_zero_release_reverse_closure(rf);return out;
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
