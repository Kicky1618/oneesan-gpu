#define BUCKET_SNAKE_REVERSE_FUSED 1
#define main pattern10_depth8_driver_main_unused
#include "../../b300/oneesan_cuda_gridfp_b300_bucket_snake_atomic.cu"
#undef main
#include "../ramstream32_bucket_closure_pattern10_depth8.hpp"
#include "../ramstream32_bucket_orbit_closure_fused.cuh"

#include <cstdint>
#include <cstdlib>
#include <iostream>

static uint64_t p10d8_nonzero(const std::vector<uint8_t>&v){uint64_t n=0;for(uint8_t x:v)n+=x!=0;return n;}
static uint8_t p10d8_max(const std::vector<uint8_t>&v){uint8_t m=0;for(uint8_t x:v)m=std::max(m,x);return m;}
static uint8_t p10d8_low_record_depth(const BucketFusedHost&f,uint32_t rid){if(rid==BKOC_NONE)return 0;const auto&r=f.low_dst[rid];uint32_t cc=r.counts>>16;if(cc>1){std::cerr<<"depth8 LOW record has multiple CROSS active sources rid="<<rid<<" count="<<cc<<'\n';std::exit(572);}return cc?uint8_t(bkf_cross_depth(f.low_cross_op[r.cross_begin])):0;}
static uint8_t p10d8_high_record_depth(const BucketFusedHost&f,uint32_t rid){if(rid==BKOC_NONE)return 0;const auto&r=f.high_dst[rid];uint32_t cc=r.counts>>16;if(cc>1){std::cerr<<"depth8 HIGH record has multiple CROSS active sources rid="<<rid<<" count="<<cc<<'\n';std::exit(573);}return cc?uint8_t(bkf_cross_depth(f.high_cross_op[r.cross_begin])):0;}

int main(){
    build_full_dp();G_FACTOR=build_factor_tables();StorageFactorHost storage=build_storage_factor_tables(G_FACTOR);StorageLayout layout=build_storage_layout(storage);
    LowDescHost lowdesc=build_low_descriptors(storage,layout);HighDescHost highdesc=build_high_descriptors(storage,layout);LowOrbitHost loworbit=build_cpu_low_orbit(storage,layout,lowdesc);CpuHighDirectHost highdirect=build_cpu_high_direct(storage,layout,highdesc);GpuDirectGatherHost ordinary=build_gpu_direct_gather(layout,lowdesc,loworbit,highdirect);GpuDirectCrossGatherHost cross=build_gpu_direct_cross_gather(storage,layout,lowdesc,loworbit,highdirect);GpuDirectFusedHost fused=build_gpu_direct_fused_checked(layout,ordinary,cross);CpuLowSparseHost lowsparse=build_cpu_low_sparse(storage,layout,lowdesc,loworbit);BucketOwnerHost owner=build_bucket_owners(G_FACTOR,storage);BucketOrbitStreamsHost bo=build_bucket_orbits(storage,layout,owner,lowsparse,highdirect);BucketFusedHost bf=build_bucket_fused(storage,layout,owner,ordinary,cross,fused);
    ReverseLowDescHost rlow=build_reverse_low_descriptors(storage,layout);ReverseHighDescHost rhigh=build_reverse_high_descriptors(storage,layout);ReverseOrbitHost rlo=build_reverse_orbit(storage,layout,true),rhi=build_reverse_orbit(storage,layout,false);ReverseBucketAtomicHost rb=build_reverse_bucket_atomic(storage,layout,owner,rlow,rhigh,rlo,rhi);ReverseBucketFusedHost rf=build_reverse_bucket_fused_checked(layout,owner,rb);
    BucketForwardOrbitClosureAttachHost fa=build_bucket_forward_orbit_closure_attach(layout,bo,bf);BucketReverseOrbitClosureAttachHost ra=build_bucket_reverse_orbit_closure_attach(layout,rb,rf);
    ReverseSplit54Host rs=build_reverse_split54(layout,rb,false);build_bucket_forward_pattern10(layout,bo,bf);build_reverse_split54_pattern10(layout,bf,rs);BucketPattern10Depth8Host d=build_bucket_pattern10_depth8(layout,bf,bo,rs);
    uint64_t expected_ops=bo.low_nn.size()+bo.low_nr.size()+bo.low_nl.size()+bo.high_nn.size()+bo.high_nrnl.size()+rs.low.ops()+rs.high.ops();if(d.ops()!=expected_ops){std::cerr<<"pattern10 depth8 op count mismatch got="<<d.ops()<<" expected="<<expected_ops<<'\n';return 2;}
    uint64_t nz=0;nz+=p10d8_nonzero(d.f_low_nn)+p10d8_nonzero(d.f_low_nr)+p10d8_nonzero(d.f_low_nl)+p10d8_nonzero(d.f_high_nn)+p10d8_nonzero(d.f_high_nrnl)+p10d8_nonzero(d.r_low_nn)+p10d8_nonzero(d.r_low_nr)+p10d8_nonzero(d.r_low_nl)+p10d8_nonzero(d.r_high_nn)+p10d8_nonzero(d.r_high_nr)+p10d8_nonzero(d.r_high_nl);
    uint8_t md=0;auto mx=[&](const auto&v){md=std::max(md,p10d8_max(v));};mx(d.f_low_nn);mx(d.f_low_nr);mx(d.f_low_nl);mx(d.f_high_nn);mx(d.f_high_nrnl);mx(d.r_low_nn);mx(d.r_low_nr);mx(d.r_low_nl);mx(d.r_high_nn);mx(d.r_high_nr);mx(d.r_high_nl);if(md>15)return 3;

    auto cmp=[&](uint8_t got,uint8_t want,const char*side,size_t q){if(got!=want){std::cerr<<"pattern10 depth8 descriptor mismatch side="<<side<<" q="<<q<<" got="<<unsigned(got)<<" want="<<unsigned(want)<<'\n';std::exit(574);}};
    for(size_t q=0;q<fa.low_nn.size();++q)cmp(d.f_low_nn[q],p10d8_low_record_depth(bf,fa.low_nn[q]),"forward-low-nn",q);
    for(size_t q=0;q<fa.low_nr.size();++q)cmp(d.f_low_nr[q],p10d8_low_record_depth(bf,fa.low_nr[q]),"forward-low-nr",q);
    for(size_t q=0;q<fa.low_nl.size();++q)cmp(d.f_low_nl[q],p10d8_low_record_depth(bf,fa.low_nl[q]),"forward-low-nl",q);
    for(size_t q=0;q<fa.high_nn.size();++q)cmp(d.f_high_nn[q],p10d8_high_record_depth(bf,fa.high_nn[q]),"forward-high-nn",q);
    for(size_t q=0;q<fa.high_nrnl.size();++q)cmp(d.f_high_nrnl[q],p10d8_high_record_depth(bf,fa.high_nrnl[q]),"forward-high-nrnl",q);

    // ReverseSplit54 preserves p/bid order but separates NN/NR/NL streams.
    // Replay the original reverse orbit order and advance one counter per kind.
    size_t lnn=0,lnr=0,lnl=0,hnn=0,hnr=0,hnl=0;size_t rp=size_t(rb.nblocks)+1;
    for(uint32_t pi=0;pi<uint32_t(LOW_LUT_K);++pi)for(uint32_t bid=0;bid<rb.nblocks;++bid){uint32_t a=rb.low_orbit_off[size_t(pi)*rp+bid],b=rb.low_orbit_off[size_t(pi)*rp+bid+1];for(uint32_t q=a;q<b;++q){uint32_t k=rb_orbit_kind(rb.low_orbit[q]);uint8_t want=p10d8_low_record_depth(rf,ra.low[q]);if(k==CPU_ORBIT_NN)cmp(d.r_low_nn[lnn++],want,"reverse-low-nn",q);else if(k==CPU_ORBIT_NR)cmp(d.r_low_nr[lnr++],want,"reverse-low-nr",q);else if(k==CPU_ORBIT_NL)cmp(d.r_low_nl[lnl++],want,"reverse-low-nl",q);else return 4;}}
    for(uint32_t pi=0;pi<uint32_t(HIGH_LUT_K);++pi)for(uint32_t bid=0;bid<rb.nblocks;++bid){uint32_t a=rb.high_orbit_off[size_t(pi)*rp+bid],b=rb.high_orbit_off[size_t(pi)*rp+bid+1];for(uint32_t q=a;q<b;++q){uint32_t k=rb_orbit_kind(rb.high_orbit[q]);uint8_t want=p10d8_high_record_depth(rf,ra.high[q]);if(k==CPU_ORBIT_NN)cmp(d.r_high_nn[hnn++],want,"reverse-high-nn",q);else if(k==CPU_ORBIT_NR)cmp(d.r_high_nr[hnr++],want,"reverse-high-nr",q);else if(k==CPU_ORBIT_NL)cmp(d.r_high_nl[hnl++],want,"reverse-high-nl",q);else return 5;}}
    if(lnn!=d.r_low_nn.size()||lnr!=d.r_low_nr.size()||lnl!=d.r_low_nl.size()||hnn!=d.r_high_nn.size()||hnr!=d.r_high_nr.size()||hnl!=d.r_high_nl.size())return 6;
    std::cout<<"closure-pattern10-depth8-plan OK W="<<TARGET_W<<" ops="<<d.ops()<<" nonzero_cross="<<nz<<" max_depth="<<unsigned(md)<<" sidecar_mib="<<double(d.bytes())/double(1<<20)<<" bytes_per_orbit=1 descriptor_depth_exact=1 cross_scan_eliminable=1\n";return 0;
}
