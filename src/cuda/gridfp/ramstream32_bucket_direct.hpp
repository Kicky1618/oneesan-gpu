#pragma once

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <utility>
#include <vector>

// Translate already-proven sparse/direct orbit metadata from height-local
// all-ranks into owner-local bucket locators. No topology reconstruction is
// needed here. Each locator is 3-bit owner + 15-bit owner-local rank, so the
// source/partner/drop triple occupies 54 bits of one uint64_t.

using BucketOrbitOp=uint64_t;
static constexpr uint64_t BUCKET_ORBIT_LOC_MASK=(1ull<<BUCKET_LOCATOR_BITS)-1ull;
static constexpr int BUCKET_ORBIT_PARTNER_SHIFT=BUCKET_LOCATOR_BITS;
static constexpr int BUCKET_ORBIT_DROP_SHIFT=2*BUCKET_LOCATOR_BITS;
static_assert(3*BUCKET_LOCATOR_BITS==54);

static inline BucketOrbitOp bucket_orbit_pack(uint32_t src,uint32_t partner,uint32_t drop){
    if((uint64_t(src)&~BUCKET_ORBIT_LOC_MASK)
       ||(uint64_t(partner)&~BUCKET_ORBIT_LOC_MASK)
       ||(uint64_t(drop)&~BUCKET_ORBIT_LOC_MASK)){
        std::cerr<<"bucket orbit locator overflow\n";std::exit(190);
    }
    return uint64_t(src)|(uint64_t(partner)<<BUCKET_ORBIT_PARTNER_SHIFT)
        |(uint64_t(drop)<<BUCKET_ORBIT_DROP_SHIFT);
}
static inline uint32_t bucket_orbit_src(BucketOrbitOp x){return uint32_t(x&BUCKET_ORBIT_LOC_MASK);}
static inline uint32_t bucket_orbit_partner(BucketOrbitOp x){return uint32_t((x>>BUCKET_ORBIT_PARTNER_SHIFT)&BUCKET_ORBIT_LOC_MASK);}
static inline uint32_t bucket_orbit_drop(BucketOrbitOp x){return uint32_t((x>>BUCKET_ORBIT_DROP_SHIFT)&BUCKET_ORBIT_LOC_MASK);}

struct BucketOrbitStreamsHost {
    std::vector<BucketOrbitOp> low_nn,low_nr,low_nl;
    std::vector<uint32_t> low_nn_off,low_nr_off,low_nl_off;
    std::vector<BucketOrbitOp> high_nn,high_nrnl;
    std::vector<uint32_t> high_nn_off,high_nrnl_off;
    uint32_t low_nblocks=0,high_nblocks=0;
    size_t bytes()const{
        return (low_nn.size()+low_nr.size()+low_nl.size()+high_nn.size()+high_nrnl.size())*sizeof(BucketOrbitOp)
            +(low_nn_off.size()+low_nr_off.size()+low_nl_off.size()+high_nn_off.size()+high_nrnl_off.size())*sizeof(uint32_t);
    }
};

static uint32_t bucket_low_locator(
    const StorageFactorHost& storage,const BucketOwnerHost& owner,int h,uint32_t all_rank
){
    uint32_t n=storage.low_all_off[h+1]-storage.low_all_off[h];
    if(all_rank>=n){std::cerr<<"bucket LOW all-rank overflow h="<<h<<" rank="<<all_rank<<'/'<<n<<'\n';std::exit(191);}
    return owner.low_all_locator[storage.low_all_off[h]+all_rank];
}
static uint32_t bucket_high_locator(
    const StorageFactorHost& storage,const BucketOwnerHost& owner,int h,uint32_t all_rank
){
    uint32_t n=storage.high_all_off[h+1]-storage.high_all_off[h];
    if(all_rank>=n){std::cerr<<"bucket HIGH all-rank overflow h="<<h<<" rank="<<all_rank<<'/'<<n<<'\n';std::exit(192);}
    return owner.high_all_locator[storage.high_all_off[h]+all_rank];
}

static BucketOrbitStreamsHost build_bucket_orbits(
    const StorageFactorHost& storage,const StorageLayout& layout,
    const BucketOwnerHost& owner,const CpuLowSparseHost& low,
    const CpuHighDirectHost& high
){
    BucketOrbitStreamsHost out;
    out.low_nblocks=low.nblocks;out.high_nblocks=high.nblocks;
    size_t lpitch=size_t(low.nblocks)+1;
    out.low_nn_off.resize(size_t(LOW_LUT_K)*lpitch);
    out.low_nr_off.resize(size_t(LOW_LUT_K)*lpitch);
    out.low_nl_off.resize(size_t(LOW_LUT_K)*lpitch);

    auto translate_low_stream=[&](
        uint32_t pi,uint32_t bid,int p,uint32_t kind,
        const std::vector<CpuLowSparseOrbitOp>& src,
        const std::vector<uint32_t>& off,std::vector<BucketOrbitOp>& dst,
        std::vector<uint32_t>& doff
    ){
        size_t oi=size_t(pi)*lpitch+bid;doff[oi]=uint32_t(dst.size());
        auto [a,b]=cpu_sparse_range(off,low.nblocks,pi,bid);
        const StorageBlock& xb=layout.main_blocks[bid];
        FBlock x{};x.he=xb.he;x.hs=xb.hs;x.c=xb.c;
        uint32_t jbid=cpu_sparse_jblock(bid,x,p,kind);
        uint32_t dbid=uint32_t(xb.he);
        const StorageBlock& jb=layout.main_blocks[jbid];
        const StorageBlock& db=layout.block_blocks[dbid];
        for(uint32_t q=a;q<b;++q){
            CpuLowSparseOrbitOp z=src[q];
            uint32_t sl=bucket_low_locator(storage,owner,xb.hs,cpu_sparse_src(z));
            uint32_t jl=bucket_low_locator(storage,owner,jb.hs,cpu_sparse_jlr(z));
            uint32_t dl=bucket_low_locator(storage,owner,db.hs,cpu_sparse_dlr(z));
            dst.push_back(bucket_orbit_pack(sl,jl,dl));
        }
    };
    for(int p=LOW_LUT_K;p>=1;--p){
        uint32_t pi=uint32_t(LOW_LUT_K-p);
        for(uint32_t bid=0;bid<low.nblocks;++bid){
            translate_low_stream(pi,bid,p,CPU_ORBIT_NN,low.nn_orbit_ops,low.nn_orbit_off,out.low_nn,out.low_nn_off);
            translate_low_stream(pi,bid,p,CPU_ORBIT_NR,low.nr_orbit_ops,low.nr_orbit_off,out.low_nr,out.low_nr_off);
            translate_low_stream(pi,bid,p,CPU_ORBIT_NL,low.nl_orbit_ops,low.nl_orbit_off,out.low_nl,out.low_nl_off);
        }
        size_t end=size_t(pi)*lpitch+low.nblocks;
        out.low_nn_off[end]=uint32_t(out.low_nn.size());
        out.low_nr_off[end]=uint32_t(out.low_nr.size());
        out.low_nl_off[end]=uint32_t(out.low_nl.size());
    }

    size_t hpitch=size_t(high.nblocks)+1;
    out.high_nn_off.resize(size_t(HIGH_LUT_K)*hpitch);
    out.high_nrnl_off.resize(size_t(HIGH_LUT_K)*hpitch);
    auto translate_high_stream=[&](
        uint32_t pi,uint32_t bid,int p,bool nn,
        const std::vector<CpuHighOrbitOp>& src,const std::vector<uint32_t>& off,
        std::vector<BucketOrbitOp>& dst,std::vector<uint32_t>& doff
    ){
        size_t oi=size_t(pi)*hpitch+bid;doff[oi]=uint32_t(dst.size());
        auto [a,b]=cpu_high_direct_range(off,high.nblocks,pi,bid);
        const StorageBlock& xb=layout.main_blocks[bid];
        FBlock x{};x.he=xb.he;x.hs=xb.hs;x.c=xb.c;
        uint32_t jbid=cpu_high_orbit_partner_block(bid,x,p,nn);
        uint32_t dbid=cpu_high_orbit_drop_block(x);
        const StorageBlock& jb=layout.main_blocks[jbid];
        const StorageBlock& db=layout.block_blocks[dbid];
        for(uint32_t q=a;q<b;++q){
            CpuHighOrbitOp z=src[q];
            uint32_t sl=bucket_high_locator(storage,owner,xb.he,cpu_high_orbit_src(z));
            uint32_t jl=bucket_high_locator(storage,owner,jb.he,cpu_high_orbit_partner(z));
            uint32_t dl=bucket_high_locator(storage,owner,db.he,cpu_high_orbit_drop(z));
            dst.push_back(bucket_orbit_pack(sl,jl,dl));
        }
    };
    for(int p=TARGET_W-1;p>=LOW_LUT_K+1;--p){
        uint32_t pi=uint32_t((TARGET_W-1)-p);
        for(uint32_t bid=0;bid<high.nblocks;++bid){
            translate_high_stream(pi,bid,p,true,high.orbit_ops.nn,high.orbit_off.nn,out.high_nn,out.high_nn_off);
            translate_high_stream(pi,bid,p,false,high.orbit_ops.nrnl,high.orbit_off.nrnl,out.high_nrnl,out.high_nrnl_off);
        }
        size_t end=size_t(pi)*hpitch+high.nblocks;
        out.high_nn_off[end]=uint32_t(out.high_nn.size());
        out.high_nrnl_off[end]=uint32_t(out.high_nrnl.size());
    }

    if(out.low_nn.size()!=low.nn_orbit_ops.size()
       ||out.low_nr.size()!=low.nr_orbit_ops.size()
       ||out.low_nl.size()!=low.nl_orbit_ops.size()
       ||out.high_nn.size()!=high.orbit_ops.nn.size()
       ||out.high_nrnl.size()!=high.orbit_ops.nrnl.size()){
        std::cerr<<"bucket orbit stream size mismatch\n";std::exit(193);
    }
    std::cerr<<"bucket_orbits low="<<(out.low_nn.size()+out.low_nr.size()+out.low_nl.size())
             <<" high="<<(out.high_nn.size()+out.high_nrnl.size())
             <<" mib="<<double(out.bytes())/double(1<<20)<<'\n';
    return out;
}
