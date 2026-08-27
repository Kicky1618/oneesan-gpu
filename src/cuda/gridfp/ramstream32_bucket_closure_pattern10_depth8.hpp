#pragma once

#include "ramstream32_bucket_closure_pattern10.cuh"
#include "ramstream32_bucket_reverse_split54.cuh"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

// Pattern10 removes ordinary-closure scans without per-orbit storage.  The
// remaining CROSS depth scan is also topology-only.  Store just that depth as
// one byte per orbit for the experimental depth8 backend; 0 means no CROSS.
// A later nibble-packed variant can halve this storage if the byte load proves
// worthwhile, but uint8_t keeps the first implementation coalesced/simple.
struct BucketPattern10Depth8Host {
    std::vector<uint8_t> f_low_nn,f_low_nr,f_low_nl,f_high_nn,f_high_nrnl;
    std::vector<uint8_t> r_low_nn,r_low_nr,r_low_nl,r_high_nn,r_high_nr,r_high_nl;
    size_t bytes()const{
        return f_low_nn.size()+f_low_nr.size()+f_low_nl.size()+f_high_nn.size()+f_high_nrnl.size()
            +r_low_nn.size()+r_low_nr.size()+r_low_nl.size()+r_high_nn.size()+r_high_nr.size()+r_high_nl.size();
    }
    size_t ops()const{return bytes();}
};

static uint8_t bkcp10_depth8_low_host(MateID d,int p,const char*what){
    MateID source=0;int depth=oneesan::gridfp::low_cross_preimage_partial(d,LOW_LUT_K+1,p,source);
    if(depth<0||depth>15){std::cerr<<"pattern10 depth8 LOW overflow "<<what<<" p="<<p<<" depth="<<depth<<'\n';std::exit(570);}return uint8_t(depth);
}
static uint8_t bkcp10_depth8_high_host(MateID d,int rel,const char*what){
    MateID source=0;int depth=oneesan::gridfp::high_cross_preimage_partial(d,HIGH_LUT_K+1,rel,source);
    if(depth<0||depth>15){std::cerr<<"pattern10 depth8 HIGH overflow "<<what<<" rel="<<rel<<" depth="<<depth<<'\n';std::exit(571);}return uint8_t(depth);
}

static BucketPattern10Depth8Host build_bucket_pattern10_depth8(
    const StorageLayout&layout,const BucketFusedHost&bf,
    const BucketOrbitStreamsHost&bo,const ReverseSplit54Host&rs
){
    BucketPattern10Depth8Host out;
    auto sized=[&](auto&d,const auto&s){d.resize(s.size(),0);};
    sized(out.f_low_nn,bo.low_nn);sized(out.f_low_nr,bo.low_nr);sized(out.f_low_nl,bo.low_nl);
    sized(out.f_high_nn,bo.high_nn);sized(out.f_high_nrnl,bo.high_nrnl);
    sized(out.r_low_nn,rs.low.nn);sized(out.r_low_nr,rs.low.nr);sized(out.r_low_nl,rs.low.nl);
    sized(out.r_high_nn,rs.high.nn);sized(out.r_high_nr,rs.high.nr);sized(out.r_high_nl,rs.high.nl);
    uint64_t nz=0;uint8_t max_depth=0;
    auto put=[&](std::vector<uint8_t>&v,uint32_t q,uint8_t d){v[q]=d;nz+=d!=0;max_depth=std::max(max_depth,d);};

    size_t lp=size_t(bo.low_nblocks)+1,hp=size_t(bo.high_nblocks)+1;
    for(int p=LOW_LUT_K;p>=1;--p){uint32_t pi=uint32_t(LOW_LUT_K-p);for(uint32_t bid=0;bid<bo.low_nblocks;++bid){const auto&xb=layout.main_blocks[bid];if(!xb.valid)continue;const auto&db=p==1?xb:layout.block_blocks[xb.he];
        auto scan=[&](const std::vector<BucketOrbitOp>&ops,const std::vector<uint32_t>&off,std::vector<uint8_t>&dst,bool active){uint32_t a=off[size_t(pi)*lp+bid],b=off[size_t(pi)*lp+bid+1];for(uint32_t q=a;q<b;++q){uint8_t dep=0;if(active){uint32_t loc=p==1?bkf_orbit_src(ops[q]):bkf_orbit_drop(ops[q]);uint32_t dc=bkcp10_low_code_host(bf,loc,db.hs);MateID d=p==1?(MateID(dc)|(MateID(db.c)<<(2*LOW_LUT_K))):minsert(MateID(dc),p,N);dep=bkcp10_depth8_low_host(d,p,"forward-low");}put(dst,q,dep);}};
        scan(bo.low_nn,bo.low_nn_off,out.f_low_nn,true);scan(bo.low_nr,bo.low_nr_off,out.f_low_nr,p!=1);scan(bo.low_nl,bo.low_nl_off,out.f_low_nl,p!=1);
    }}
    for(int p=TARGET_W-1;p>=LOW_LUT_K+1;--p){uint32_t pi=uint32_t((TARGET_W-1)-p);int rel=p-LOW_LUT_K;for(uint32_t bid=0;bid<bo.high_nblocks;++bid){const auto&xb=layout.main_blocks[bid];if(!xb.valid)continue;const auto&db=layout.block_blocks[xb.hs];
        auto scan=[&](const std::vector<BucketOrbitOp>&ops,const std::vector<uint32_t>&off,std::vector<uint8_t>&dst){uint32_t a=off[size_t(pi)*hp+bid],b=off[size_t(pi)*hp+bid+1];for(uint32_t q=a;q<b;++q){uint32_t loc=bkf_orbit_drop(ops[q]),dc=bkcp10_high_code_host(bf,loc,db.he);MateID d=minsert(MateID(dc),rel,N);put(dst,q,bkcp10_depth8_high_host(d,rel,"forward-high"));}};
        scan(bo.high_nn,bo.high_nn_off,out.f_high_nn);scan(bo.high_nrnl,bo.high_nrnl_off,out.f_high_nrnl);
    }}

    size_t rp=size_t(rs.nblocks)+1;
    for(int p=1;p<=LOW_LUT_K;++p){uint32_t pi=uint32_t(p-1);for(uint32_t bid=0;bid<rs.nblocks;++bid){const auto&xb=layout.main_blocks[bid];if(!xb.valid)continue;const auto&db=layout.block_blocks[xb.he];
        auto scan=[&](const std::vector<BucketOrbitOp>&ops,const std::vector<uint32_t>&off,std::vector<uint8_t>&dst){uint32_t a=off[size_t(pi)*rp+bid],b=off[size_t(pi)*rp+bid+1];for(uint32_t q=a;q<b;++q){uint32_t loc=bkf_orbit_drop(ops[q]),dc=bkcp10_low_code_host(bf,loc,db.hs);MateID d=blocked_exclude_reverse(MateID(dc),LOW_LUT_K+1,p);put(dst,q,bkcp10_depth8_low_host(d,p,"reverse-low"));}};
        scan(rs.low.nn,rs.low.nn_off,out.r_low_nn);scan(rs.low.nr,rs.low.nr_off,out.r_low_nr);scan(rs.low.nl,rs.low.nl_off,out.r_low_nl);
    }}
    for(int p=LOW_LUT_K+1;p<TARGET_W;++p){uint32_t pi=uint32_t(p-(LOW_LUT_K+1));int rel=p-LOW_LUT_K;bool edge=p==TARGET_W-1;for(uint32_t bid=0;bid<rs.nblocks;++bid){const auto&xb=layout.main_blocks[bid];if(!xb.valid)continue;const auto&db=edge?xb:layout.block_blocks[xb.hs];
        auto scan=[&](const std::vector<BucketOrbitOp>&ops,const std::vector<uint32_t>&off,std::vector<uint8_t>&dst,bool nn){uint32_t a=off[size_t(pi)*rp+bid],b=off[size_t(pi)*rp+bid+1];for(uint32_t q=a;q<b;++q){uint8_t dep=0;if(!edge||nn){uint32_t loc=edge?bkf_orbit_src(ops[q]):bkf_orbit_drop(ops[q]);uint32_t dc=bkcp10_high_code_host(bf,loc,db.he);MateID d=edge?(MateID(db.c)|(MateID(dc)<<2)):blocked_exclude_reverse(MateID(dc),HIGH_LUT_K+1,rel);dep=bkcp10_depth8_high_host(d,rel,"reverse-high");}put(dst,q,dep);}};
        scan(rs.high.nn,rs.high.nn_off,out.r_high_nn,true);scan(rs.high.nr,rs.high.nr_off,out.r_high_nr,false);scan(rs.high.nl,rs.high.nl_off,out.r_high_nl,false);
    }}
    std::cerr<<"bucket_pattern10_depth8 ops="<<out.ops()<<" nonzero_cross="<<nz<<" max_depth="<<unsigned(max_depth)<<" bytes_per_orbit=1 mib="<<double(out.bytes())/double(1<<20)<<'\n';
    return out;
}
