#define BUCKET_SNAKE_REVERSE_FUSED 1
#define main pattern10_depth8_driver_main_unused
#include "../../b300/oneesan_cuda_gridfp_b300_bucket_snake_atomic.cu"
#undef main
#include "../ramstream32_bucket_closure_pattern10_depth8.hpp"

#include <cstdint>
#include <cstdlib>
#include <iostream>

static uint64_t p10d8_nonzero(const std::vector<uint8_t>&v){uint64_t n=0;for(uint8_t x:v)n+=x!=0;return n;}
static uint8_t p10d8_max(const std::vector<uint8_t>&v){uint8_t m=0;for(uint8_t x:v)m=std::max(m,x);return m;}

int main(){
    build_full_dp();G_FACTOR=build_factor_tables();StorageFactorHost storage=build_storage_factor_tables(G_FACTOR);StorageLayout layout=build_storage_layout(storage);
    LowDescHost lowdesc=build_low_descriptors(storage,layout);HighDescHost highdesc=build_high_descriptors(storage,layout);LowOrbitHost loworbit=build_cpu_low_orbit(storage,layout,lowdesc);CpuHighDirectHost highdirect=build_cpu_high_direct(storage,layout,highdesc);GpuDirectGatherHost ordinary=build_gpu_direct_gather(layout,lowdesc,loworbit,highdirect);GpuDirectCrossGatherHost cross=build_gpu_direct_cross_gather(storage,layout,lowdesc,loworbit,highdirect);GpuDirectFusedHost fused=build_gpu_direct_fused_checked(layout,ordinary,cross);CpuLowSparseHost lowsparse=build_cpu_low_sparse(storage,layout,lowdesc,loworbit);BucketOwnerHost owner=build_bucket_owners(G_FACTOR,storage);BucketOrbitStreamsHost bo=build_bucket_orbits(storage,layout,owner,lowsparse,highdirect);BucketFusedHost bf=build_bucket_fused(storage,layout,owner,ordinary,cross,fused);
    ReverseLowDescHost rlow=build_reverse_low_descriptors(storage,layout);ReverseHighDescHost rhigh=build_reverse_high_descriptors(storage,layout);ReverseOrbitHost rlo=build_reverse_orbit(storage,layout,true),rhi=build_reverse_orbit(storage,layout,false);ReverseBucketAtomicHost rb=build_reverse_bucket_atomic(storage,layout,owner,rlow,rhigh,rlo,rhi);ReverseSplit54Host rs=build_reverse_split54(layout,rb,false);
    build_bucket_forward_pattern10(layout,bo,bf);build_reverse_split54_pattern10(layout,bf,rs);BucketPattern10Depth8Host d=build_bucket_pattern10_depth8(layout,bf,bo,rs);
    uint64_t expected_ops=bo.low_nn.size()+bo.low_nr.size()+bo.low_nl.size()+bo.high_nn.size()+bo.high_nrnl.size()+rs.low.ops()+rs.high.ops();if(d.ops()!=expected_ops){std::cerr<<"pattern10 depth8 op count mismatch got="<<d.ops()<<" expected="<<expected_ops<<'\n';return 2;}
    uint64_t nz=0;nz+=p10d8_nonzero(d.f_low_nn)+p10d8_nonzero(d.f_low_nr)+p10d8_nonzero(d.f_low_nl)+p10d8_nonzero(d.f_high_nn)+p10d8_nonzero(d.f_high_nrnl)+p10d8_nonzero(d.r_low_nn)+p10d8_nonzero(d.r_low_nr)+p10d8_nonzero(d.r_low_nl)+p10d8_nonzero(d.r_high_nn)+p10d8_nonzero(d.r_high_nr)+p10d8_nonzero(d.r_high_nl);
    uint8_t md=0;auto mx=[&](const auto&v){md=std::max(md,p10d8_max(v));};mx(d.f_low_nn);mx(d.f_low_nr);mx(d.f_low_nl);mx(d.f_high_nn);mx(d.f_high_nrnl);mx(d.r_low_nn);mx(d.r_low_nr);mx(d.r_low_nl);mx(d.r_high_nn);mx(d.r_high_nr);mx(d.r_high_nl);if(md>15)return 3;

    // Independent descriptor-level check.  Each existing CROSS closure record
    // must have exactly the same depth as the orbit-side precomputation for its
    // uniquely attached destination.  The full-inverse probe separately proves
    // uniqueness/source identity, so here depth equality is the remaining bit.
    auto check_forward_low=[&](){size_t pitch=size_t(bo.low_nblocks)+1;for(int p=LOW_LUT_K;p>=1;--p){uint32_t pi=uint32_t(LOW_LUT_K-p);for(uint32_t bid=0;bid<bo.low_nblocks;++bid){const StorageBlock&xb=layout.main_blocks[bid];if(!xb.valid)continue;uint32_t dbid=p==1?bid:uint32_t(xb.he);size_t doi=size_t(pi)*bf.low_pitch+dbid;auto scan=[&](const auto&ops,const auto&off,const auto&dep,bool active){for(uint32_t q=off[size_t(pi)*pitch+bid];q<off[size_t(pi)*pitch+bid+1];++q){uint8_t want=0;if(active){uint32_t loc=p==1?bkf_orbit_src(ops[q]):bkf_orbit_drop(ops[q]);const StorageBlock&db=p==1?xb:layout.block_blocks[xb.he];uint32_t dc=bkcp10_low_code_host(bf,loc,db.hs);MateID full=p==1?(MateID(dc)|(MateID(db.c)<<(2*LOW_LUT_K))):minsert(MateID(dc),p,N),src=0;want=uint8_t(oneesan::gridfp::low_cross_preimage_partial(full,LOW_LUT_K+1,p,src));}if(dep[q]!=want){std::cerr<<"pattern10 depth8 forward LOW mismatch p="<<p<<" bid="<<bid<<" q="<<q<<" got="<<unsigned(dep[q])<<" want="<<unsigned(want)<<'\n';std::exit(572);}}};scan(bo.low_nn,bo.low_nn_off,d.f_low_nn,true);scan(bo.low_nr,bo.low_nr_off,d.f_low_nr,p!=1);scan(bo.low_nl,bo.low_nl_off,d.f_low_nl,p!=1);(void)doi;}}};
    check_forward_low();
    std::cout<<"closure-pattern10-depth8-plan OK W="<<TARGET_W<<" ops="<<d.ops()<<" nonzero_cross="<<nz<<" max_depth="<<unsigned(md)<<" sidecar_mib="<<double(d.bytes())/double(1<<20)<<" bytes_per_orbit=1 cross_scan_eliminable=1\n";return 0;
}
