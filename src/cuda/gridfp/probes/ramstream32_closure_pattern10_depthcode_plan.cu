#define BUCKET_SNAKE_REVERSE_FUSED 1
#define main pattern10_depthcode_driver_main_unused
#include "../../b300/oneesan_cuda_gridfp_b300_bucket_snake_atomic.cu"
#undef main
#include "../ramstream32_bucket_closure_pattern10_depth8.hpp"

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <set>
#include <vector>

int main(){
    build_full_dp();G_FACTOR=build_factor_tables();StorageFactorHost storage=build_storage_factor_tables(G_FACTOR);StorageLayout layout=build_storage_layout(storage);
    LowDescHost lowdesc=build_low_descriptors(storage,layout);HighDescHost highdesc=build_high_descriptors(storage,layout);LowOrbitHost loworbit=build_cpu_low_orbit(storage,layout,lowdesc);CpuHighDirectHost highdirect=build_cpu_high_direct(storage,layout,highdesc);GpuDirectGatherHost ordinary=build_gpu_direct_gather(layout,lowdesc,loworbit,highdirect);GpuDirectCrossGatherHost cross=build_gpu_direct_cross_gather(storage,layout,lowdesc,loworbit,highdirect);GpuDirectFusedHost fused=build_gpu_direct_fused_checked(layout,ordinary,cross);CpuLowSparseHost lowsparse=build_cpu_low_sparse(storage,layout,lowdesc,loworbit);BucketOwnerHost owner=build_bucket_owners(G_FACTOR,storage);BucketOrbitStreamsHost bo=build_bucket_orbits(storage,layout,owner,lowsparse,highdirect);BucketFusedHost bf=build_bucket_fused(storage,layout,owner,ordinary,cross,fused);
    ReverseLowDescHost rlow=build_reverse_low_descriptors(storage,layout);ReverseHighDescHost rhigh=build_reverse_high_descriptors(storage,layout);ReverseOrbitHost rlo=build_reverse_orbit(storage,layout,true),rhi=build_reverse_orbit(storage,layout,false);ReverseBucketAtomicHost rb=build_reverse_bucket_atomic(storage,layout,owner,rlow,rhigh,rlo,rhi);ReverseSplit54Host rs=build_reverse_split54(layout,rb,false);
    build_bucket_forward_pattern10(layout,bo,bf);build_reverse_split54_pattern10(layout,bf,rs);BucketPattern10Depth8Host d=build_bucket_pattern10_depth8(layout,bf,bo,rs);

    constexpr uint32_t LH=HIGH_LUT_K+2,HH=LOW_LUT_K+2;
    std::vector<std::set<uint16_t>> low(size_t(LOW_LUT_K)*LH),high(size_t(HIGH_LUT_K)*HH);
    uint64_t inserted=0;
    auto pair_code=[](BucketOrbitOp op,uint8_t depth){return uint16_t((uint16_t(bkcp10_id(op))<<4)|uint16_t(depth));};
    auto add_low=[&](int p,uint32_t h,BucketOrbitOp op,uint8_t depth){if(h>=LH){std::cerr<<"depthcode low height overflow h="<<h<<'\n';std::exit(580);}low[size_t(p-1)*LH+h].insert(pair_code(op,depth));++inserted;};
    auto add_high=[&](int p,uint32_t h,BucketOrbitOp op,uint8_t depth){if(h>=HH){std::cerr<<"depthcode high height overflow h="<<h<<'\n';std::exit(581);}high[size_t(p-(LOW_LUT_K+1))*HH+h].insert(pair_code(op,depth));++inserted;};

    size_t lp=size_t(bo.low_nblocks)+1,hp=size_t(bo.high_nblocks)+1;
    for(int p=LOW_LUT_K;p>=1;--p){uint32_t pi=uint32_t(LOW_LUT_K-p);for(uint32_t bid=0;bid<bo.low_nblocks;++bid){const auto&xb=layout.main_blocks[bid];if(!xb.valid)continue;auto scan=[&](const auto&ops,const auto&off,const auto&dep){for(uint32_t q=off[size_t(pi)*lp+bid];q<off[size_t(pi)*lp+bid+1];++q)add_low(p,xb.he,ops[q],dep[q]);};scan(bo.low_nn,bo.low_nn_off,d.f_low_nn);scan(bo.low_nr,bo.low_nr_off,d.f_low_nr);scan(bo.low_nl,bo.low_nl_off,d.f_low_nl);}}
    for(int p=TARGET_W-1;p>=LOW_LUT_K+1;--p){uint32_t pi=uint32_t((TARGET_W-1)-p);for(uint32_t bid=0;bid<bo.high_nblocks;++bid){const auto&xb=layout.main_blocks[bid];if(!xb.valid)continue;auto scan=[&](const auto&ops,const auto&off,const auto&dep){for(uint32_t q=off[size_t(pi)*hp+bid];q<off[size_t(pi)*hp+bid+1];++q)add_high(p,xb.hs,ops[q],dep[q]);};scan(bo.high_nn,bo.high_nn_off,d.f_high_nn);scan(bo.high_nrnl,bo.high_nrnl_off,d.f_high_nrnl);}}

    size_t rp=size_t(rs.nblocks)+1;
    for(int p=1;p<=LOW_LUT_K;++p){uint32_t pi=uint32_t(p-1);for(uint32_t bid=0;bid<rs.nblocks;++bid){const auto&xb=layout.main_blocks[bid];if(!xb.valid)continue;auto scan=[&](const auto&ops,const auto&off,const auto&dep){for(uint32_t q=off[size_t(pi)*rp+bid];q<off[size_t(pi)*rp+bid+1];++q)add_low(p,xb.he,ops[q],dep[q]);};scan(rs.low.nn,rs.low.nn_off,d.r_low_nn);scan(rs.low.nr,rs.low.nr_off,d.r_low_nr);scan(rs.low.nl,rs.low.nl_off,d.r_low_nl);}}
    for(int p=LOW_LUT_K+1;p<TARGET_W;++p){uint32_t pi=uint32_t(p-(LOW_LUT_K+1));for(uint32_t bid=0;bid<rs.nblocks;++bid){const auto&xb=layout.main_blocks[bid];if(!xb.valid)continue;auto scan=[&](const auto&ops,const auto&off,const auto&dep){for(uint32_t q=off[size_t(pi)*rp+bid];q<off[size_t(pi)*rp+bid+1];++q)add_high(p,xb.hs,ops[q],dep[q]);};scan(rs.high.nn,rs.high.nn_off,d.r_high_nn);scan(rs.high.nr,rs.high.nr_off,d.r_high_nr);scan(rs.high.nl,rs.high.nl_off,d.r_high_nl);}}

    size_t max_pairs=0,total_pairs=0,contexts=0;int worst_p=0,worst_h=0;const char*worst_side="none";
    for(int p=1;p<=LOW_LUT_K;++p)for(uint32_t h=0;h<LH;++h){const auto&s=low[size_t(p-1)*LH+h];if(s.empty())continue;++contexts;total_pairs+=s.size();if(s.size()>max_pairs){max_pairs=s.size();worst_p=p;worst_h=int(h);worst_side="low";}}
    for(int p=LOW_LUT_K+1;p<TARGET_W;++p)for(uint32_t h=0;h<HH;++h){const auto&s=high[size_t(p-(LOW_LUT_K+1))*HH+h];if(s.empty())continue;++contexts;total_pairs+=s.size();if(s.size()>max_pairs){max_pairs=s.size();worst_p=p;worst_h=int(h);worst_side="high";}}
    bool fits=max_pairs<=1024;size_t dense_decode_bytes=contexts*1024ull*sizeof(uint16_t);size_t compact_decode_bytes=total_pairs*sizeof(uint16_t);double sidecar_mib=double(d.bytes())/double(1<<20);
    std::cout<<std::setprecision(12)
        <<"closure-pattern10-depthcode-plan OK W="<<TARGET_W
        <<" ops="<<d.ops()
        <<" inserted="<<inserted
        <<" contexts="<<contexts
        <<" unique_pairs="<<total_pairs
        <<" max_pairs_per_context="<<max_pairs
        <<" worst_side="<<worst_side
        <<" worst_p="<<worst_p
        <<" worst_height="<<worst_h
        <<" fits10="<<(fits?1:0)
        <<" dense_decode_mib="<<double(dense_decode_bytes)/double(1<<20)
        <<" compact_decode_mib="<<double(compact_decode_bytes)/double(1<<20)
        <<" depth8_sidecar_mib="<<sidecar_mib
        <<" sidecar_bytes_per_orbit=1\n";
    return 0;
}
