#define BUCKET_SNAKE_REVERSE_FUSED 1
#define main rs18_direct_driver_main_unused
#include "../../b300/oneesan_cuda_gridfp_b300_bucket_snake_atomic.cu"
#undef main
#include "../ramstream32_bucket_reverse_fused_validate.hpp"
#include "../ramstream32_bucket_orbit_closure_preflight.cuh"
#include "../ramstream32_bucket_reverse_split18.hpp"
#include "../ramstream32_bucket_reverse_split18_direct.hpp"

#include <cstdlib>
#include <iostream>

static bool same_side(const ReverseSplit18SideHost&a,const ReverseSplit18SideHost&b){
    return a.nn==b.nn&&a.nr==b.nr&&a.nl==b.nl
        &&a.nn_hi==b.nn_hi&&a.nr_hi==b.nr_hi&&a.nl_hi==b.nl_hi
        &&a.nn_off==b.nn_off&&a.nr_off==b.nr_off&&a.nl_off==b.nl_off;
}

int main(){
    build_full_dp();
    G_FACTOR=build_factor_tables();
    StorageFactorHost storage=build_storage_factor_tables(G_FACTOR);
    StorageLayout layout=build_storage_layout(storage);
    LowDescHost lowdesc=build_low_descriptors(storage,layout);
    HighDescHost highdesc=build_high_descriptors(storage,layout);
    LowOrbitHost loworbit=build_cpu_low_orbit(storage,layout,lowdesc);
    CpuHighDirectHost highdirect=build_cpu_high_direct(storage,layout,highdesc);
    GpuDirectGatherHost ordinary=build_gpu_direct_gather(layout,lowdesc,loworbit,highdirect);
    GpuDirectCrossGatherHost cross=build_gpu_direct_cross_gather(storage,layout,lowdesc,loworbit,highdirect);
    GpuDirectFusedHost fused=build_gpu_direct_fused_checked(layout,ordinary,cross);
    CpuLowSparseHost lowsparse=build_cpu_low_sparse(storage,layout,lowdesc,loworbit);
    BucketOwnerHost owner=build_bucket_owners(G_FACTOR,storage);
    BucketOrbitStreamsHost bo=build_bucket_orbits(storage,layout,owner,lowsparse,highdirect);
    BucketFusedHost bf=build_bucket_fused(storage,layout,owner,ordinary,cross,fused);
    ReverseLowDescHost rlow=build_reverse_low_descriptors(storage,layout);
    ReverseHighDescHost rhigh=build_reverse_high_descriptors(storage,layout);
    ReverseOrbitHost rlo=build_reverse_orbit(storage,layout,true),rhi=build_reverse_orbit(storage,layout,false);
    ReverseBucketAtomicHost rb=build_reverse_bucket_atomic(storage,layout,owner,rlow,rhigh,rlo,rhi);
    ReverseBucketFusedHost rf=build_reverse_bucket_fused_checked(layout,owner,rb);

    auto att=build_bucket_reverse_orbit_closure_attach_checked(layout,bo,bf,rb,rf);
    auto legacy=build_reverse_split18(layout,rb,rf,att);
    ReverseBucketAtomicHost direct_rb=rb;
    const size_t legacy_ops=rb.low_orbit.size()+rb.high_orbit.size();
    auto direct=build_reverse_split18_direct_checked(layout,bo,bf,direct_rb,rf,true);

    if(legacy.nblocks!=direct.nblocks||!same_side(legacy.low,direct.low)||!same_side(legacy.high,direct.high)){
        std::cerr<<"reverse split18 direct output mismatch\n";
        return 10;
    }
    if(!direct_rb.low_orbit.empty()||!direct_rb.high_orbit.empty()
       ||!direct_rb.low_orbit_off.empty()||!direct_rb.high_orbit_off.empty()
       ||!direct_rb.low_closure.empty()||!direct_rb.high_closure.empty()
       ||!direct_rb.low_closure_off.empty()||!direct_rb.high_closure_off.empty()){
        std::cerr<<"reverse split18 direct legacy payload not released\n";
        return 11;
    }
    std::cout<<"reverse-split18-direct-plan OK W="<<TARGET_W
             <<" ops="<<legacy_ops
             <<" avoided_attach_bytes="<<(4ull*legacy_ops)
             <<" output_bytes="<<direct.bytes()
             <<" legacy_payload_released=1\n";
    return 0;
}
