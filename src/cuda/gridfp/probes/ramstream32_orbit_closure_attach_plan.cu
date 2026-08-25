#define BUCKET_SNAKE_REVERSE_FUSED 1
#define main bkoc_driver_main_unused
#include "../../b300/oneesan_cuda_gridfp_b300_bucket_snake_atomic.cu"
#undef main
#include "../ramstream32_bucket_reverse_fused_validate.hpp"
#include "../ramstream32_bucket_orbit_closure_fused.cuh"

int main(){
    build_full_dp();G_FACTOR=build_factor_tables();StorageFactorHost storage=build_storage_factor_tables(G_FACTOR);StorageLayout layout=build_storage_layout(storage);
    LowDescHost lowdesc=build_low_descriptors(storage,layout);HighDescHost highdesc=build_high_descriptors(storage,layout);LowOrbitHost loworbit=build_cpu_low_orbit(storage,layout,lowdesc);CpuHighDirectHost highdirect=build_cpu_high_direct(storage,layout,highdesc);GpuDirectGatherHost ordinary=build_gpu_direct_gather(layout,lowdesc,loworbit,highdirect);GpuDirectCrossGatherHost cross=build_gpu_direct_cross_gather(storage,layout,lowdesc,loworbit,highdirect);GpuDirectFusedHost fused=build_gpu_direct_fused_checked(layout,ordinary,cross);CpuLowSparseHost lowsparse=build_cpu_low_sparse(storage,layout,lowdesc,loworbit);BucketOwnerHost owner=build_bucket_owners(G_FACTOR,storage);BucketOrbitStreamsHost bo=build_bucket_orbits(storage,layout,owner,lowsparse,highdirect);BucketFusedHost bf=build_bucket_fused(storage,layout,owner,ordinary,cross,fused);
    ReverseLowDescHost rlow=build_reverse_low_descriptors(storage,layout);ReverseHighDescHost rhigh=build_reverse_high_descriptors(storage,layout);ReverseOrbitHost rlo=build_reverse_orbit(storage,layout,true),rhi=build_reverse_orbit(storage,layout,false);ReverseBucketAtomicHost rb=build_reverse_bucket_atomic(storage,layout,owner,rlow,rhigh,rlo,rhi);ReverseBucketFusedHost rf=build_reverse_bucket_fused_checked(layout,owner,rb);
    auto fa=build_bucket_forward_orbit_closure_attach(layout,bo,bf);auto ra=build_bucket_reverse_orbit_closure_attach(layout,rb,rf);
    size_t orbit_ops=bo.low_nn.size()+bo.low_nr.size()+bo.low_nl.size()+bo.high_nn.size()+bo.high_nrnl.size()+rb.low_orbit.size()+rb.high_orbit.size();
    size_t closure_dst=bf.low_dst.size()+bf.high_dst.size()+rf.low_dst.size()+rf.high_dst.size();
    if(fa.low_nn.size()!=bo.low_nn.size()||fa.low_nr.size()!=bo.low_nr.size()||fa.low_nl.size()!=bo.low_nl.size()||fa.high_nn.size()!=bo.high_nn.size()||fa.high_nrnl.size()!=bo.high_nrnl.size()||ra.low.size()!=rb.low_orbit.size()||ra.high.size()!=rb.high_orbit.size())return 20;
    std::cout<<"orbit-closure-attach-plan OK W="<<TARGET_W<<" orbit_ops="<<orbit_ops<<" closure_dst="<<closure_dst<<" forward_attach_mib="<<double(fa.bytes())/double(1<<20)<<" reverse_attach_mib="<<double(ra.bytes())/double(1<<20)<<" kernels_per_position=1\n";return 0;
}
