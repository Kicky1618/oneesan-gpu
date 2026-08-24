#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>

#define RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../oneesan_cuda_gridfp_ramstream32_factorized_bidesc_compact.cu"
#undef RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../ramstream32_cpu_low_sparse.hpp"
#include "../ramstream32_cpu_high.hpp"
#include "../ramstream32_cpu_high_direct.hpp"
#include "../ramstream32_gpu_direct.cuh"
#include "../ramstream32_gpu_direct_gather.cuh"
#include "../ramstream32_gpu_direct_gather_cross.cuh"
#include "../ramstream32_gpu_direct_fused.cuh"
#include "../ramstream32_bucket_layout.hpp"
#include "../ramstream32_bucket_direct.hpp"
#include "../ramstream32_bucket_fused.cuh"

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
    GpuDirectFusedHost fused=build_gpu_direct_fused(layout,ordinary,cross);
    CpuLowSparseHost lowsparse=build_cpu_low_sparse(storage,layout,lowdesc,loworbit);
    BucketOwnerHost owner=build_bucket_owners(G_FACTOR,storage);
    BucketPhysicalLayoutHost phy=build_bucket_physical_layout(layout,owner);
    BucketOrbitStreamsHost borbit=build_bucket_orbits(storage,layout,owner,lowsparse,highdirect);
    BucketFusedHost bfused=build_bucket_fused(storage,layout,owner,ordinary,cross,fused);

    Code total=layout.main_size+layout.block_size;
    Code max_slot=0,max_gpu=0,min_gpu=~Code(0);
    for(int g=0;g<BUCKET_NGPU;++g){
        max_gpu=std::max(max_gpu,phy.gpu_capacity[g]);min_gpu=std::min(min_gpu,phy.gpu_capacity[g]);
        for(int s=0;s<BUCKET_NGPU;++s)max_slot=std::max(max_slot,phy.slot_capacity[g][s]);
    }
    double auth_gib=double(total*sizeof(Count))/double(1ULL<<30);
    double max_gpu_gib=double(max_gpu*sizeof(Count))/double(1ULL<<30);
    double max_slot_gib=double(max_slot*sizeof(Count))/double(1ULL<<30);
    double meta_mib=double(borbit.bytes()+bfused.bytes())/double(1<<20);

    std::cout<<std::setprecision(15)
             <<"backend=gridfp-bucket-fused-plan"
             <<" W="<<TARGET_W
             <<" main_states="<<layout.main_size
             <<" blocked_states="<<layout.block_size
             <<" states="<<total
             <<" authoritative_gib="<<auth_gib
             <<" max_gpu_slot_gib="<<max_gpu_gib
             <<" max_single_slot_gib="<<max_slot_gib
             <<" gpu_capacity_spread_mib="<<double((max_gpu-min_gpu)*sizeof(Count))/double(1<<20)
             <<" owner_high_max="<<owner.max_high_count
             <<" owner_low_max="<<owner.max_low_count
             <<" locator_bits="<<BUCKET_LOCATOR_BITS
             <<" orbit_mib="<<double(borbit.bytes())/double(1<<20)
             <<" fused_mib="<<double(bfused.bytes())/double(1<<20)
             <<" resident_direct_metadata_mib="<<meta_mib
             <<" high_direct_mib="<<double(bfused.high_direct.size()*sizeof(uint32_t))/double(1<<20)
             <<" low_direct_mib="<<double(bfused.low_direct.size()*sizeof(uint32_t))/double(1<<20)
             <<" closure_atomic=0 scratch_bytes=0\n";

    if constexpr(TARGET_W==28){
        if(total!=520735012027ULL||owner.max_high_count!=19631u||owner.max_low_count!=30114u||BUCKET_LOCATOR_BITS!=18)return 2;
    }
    return 0;
}
