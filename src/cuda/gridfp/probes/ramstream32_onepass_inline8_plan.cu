#define BUCKET_SNAKE_REVERSE_FUSED 1
#define main inline8_plan_driver_main_unused
#include "../../b300/oneesan_cuda_gridfp_b300_bucket_snake_atomic.cu"
#undef main
#include "../ramstream32_bucket_onepass_inline8.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>

struct Inline8Stats{uint32_t max_local=0,max_cross=0;uint64_t records=0,sources=0,residual=0;};
static Inline8Stats inline8_stats(const std::vector<BucketFusedDst>&r,const std::vector<uint32_t>&l,const std::vector<uint32_t>&x){Inline8Stats s;s.records=r.size();s.sources=l.size()+x.size();for(const auto&z:r){uint32_t lc=z.counts&0xffffu,cc=z.counts>>16;s.max_local=std::max(s.max_local,lc);s.max_cross=std::max(s.max_cross,cc);if(!lc&&!cc){std::cerr<<"inline8 empty destination record\n";std::exit(490);}}if(s.sources<s.records)std::exit(491);s.residual=s.sources-s.records;return s;}
int main(){
    build_full_dp();G_FACTOR=build_factor_tables();StorageFactorHost storage=build_storage_factor_tables(G_FACTOR);StorageLayout layout=build_storage_layout(storage);LowDescHost lowdesc=build_low_descriptors(storage,layout);HighDescHost highdesc=build_high_descriptors(storage,layout);LowOrbitHost loworbit=build_cpu_low_orbit(storage,layout,lowdesc);CpuHighDirectHost highdirect=build_cpu_high_direct(storage,layout,highdesc);GpuDirectGatherHost ordinary=build_gpu_direct_gather(layout,lowdesc,loworbit,highdirect);GpuDirectCrossGatherHost cross=build_gpu_direct_cross_gather(storage,layout,lowdesc,loworbit,highdirect);GpuDirectFusedHost fused=build_gpu_direct_fused_checked(layout,ordinary,cross);BucketOwnerHost owner=build_bucket_owners(G_FACTOR,storage);BucketFusedHost bf=build_bucket_fused(storage,layout,owner,ordinary,cross,fused);ReverseLowDescHost rlow=build_reverse_low_descriptors(storage,layout);ReverseHighDescHost rhigh=build_reverse_high_descriptors(storage,layout);ReverseOrbitHost rlo=build_reverse_orbit(storage,layout,true),rhi=build_reverse_orbit(storage,layout,false);ReverseBucketAtomicHost rb=build_reverse_bucket_atomic(storage,layout,owner,rlow,rhigh,rlo,rhi);ReverseBucketFusedHost rf=build_reverse_bucket_fused_checked(layout,owner,rb);
    auto fl=inline8_stats(bf.low_dst,bf.low_local_src,bf.low_cross_op),fh=inline8_stats(bf.high_dst,bf.high_local_src,bf.high_cross_op),rl=inline8_stats(rf.low_dst,rf.low_local_src,rf.low_cross_op),rh=inline8_stats(rf.high_dst,rf.high_local_src,rf.high_cross_op);auto fi=build_bucket_forward_onepass_inline8(bf);auto ri=build_bucket_reverse_onepass_inline8(rf);
    uint32_t ml=std::max({fl.max_local,fh.max_local,rl.max_local,rh.max_local}),mc=std::max({fl.max_cross,fh.max_cross,rl.max_cross,rh.max_cross});uint64_t rec=fl.records+fh.records+rl.records+rh.records,src=fl.sources+fh.sources+rl.sources+rh.sources,res=fl.residual+fh.residual+rl.residual+rh.residual;size_t old=16ull*rec+4ull*src,nw=fi.bytes()+ri.bytes();
    std::cout<<"onepass-inline8-plan OK W="<<TARGET_W<<" records="<<rec<<" sources="<<src<<" residual_sources="<<res<<" max_local="<<ml<<" max_cross="<<mc<<" count_bits=4 source_bits=28 begin_bits=28 old_mib="<<double(old)/double(1<<20)<<" inline8_mib="<<double(nw)/double(1<<20)<<" ratio="<<(old?double(nw)/double(old):0.0)<<"\n";return 0;
}
