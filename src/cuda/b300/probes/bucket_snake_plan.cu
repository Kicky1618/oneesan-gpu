#include <cuda_runtime.h>

#include <cstdint>
#include <iomanip>
#include <iostream>

#define RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../../gridfp/oneesan_cuda_gridfp_ramstream32_factorized_bidesc_compact.cu"
#undef RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../../gridfp/ramstream32_bucket_layout.hpp"
#include "../gridfp_bucket_transpose.cuh"

int main(){
    constexpr int W=TARGET_W;
    static_assert(LOW_LUT_K+HIGH_LUT_K+1==TARGET_W);
    build_full_dp();
    G_FACTOR=build_factor_tables();
    StorageFactorHost storage=build_storage_factor_tables(G_FACTOR);
    StorageLayout layout=build_storage_layout(storage);
    BucketOwnerHost owner=build_bucket_owners(G_FACTOR,storage);
    BucketPhysicalLayoutHost phy=build_bucket_physical_layout(layout,owner);
    BucketTransposePlan plan=build_bucket_transpose_plan(phy,BUCKET_NGPU);

    uint64_t peer_bytes_per_transpose=0;
    uint64_t logical_offdiag_bytes=0;
    uint64_t diagonal_bytes=0;
    for(int a=0;a<BUCKET_NGPU;++a){
        diagonal_bytes+=plan.slot[a][a].logical_bytes;
        for(int b=0;b<BUCKET_NGPU;++b)if(a!=b)
            logical_offdiag_bytes+=plan.slot[a][b].logical_bytes;
        for(int b=a+1;b<BUCKET_NGPU;++b)
            peer_bytes_per_transpose+=2ull*plan.slot[a][b].capacity_bytes;
    }

    uint64_t standard_transposes=2ull*W-1ull;
    uint64_t snake_transposes=W;
    long double standard_peer=(long double)peer_bytes_per_transpose*standard_transposes;
    long double snake_peer=(long double)peer_bytes_per_transpose*snake_transposes;
    long double saving=1.0L-snake_peer/standard_peer;
    long double raw_state_bytes=(long double)(layout.main_size+layout.block_size)*sizeof(Count);

    std::cout<<std::setprecision(15)
             <<"backend=b300-bucket-snake-plan"
             <<" W="<<W
             <<" states="<<(layout.main_size+layout.block_size)
             <<" authoritative_tib="<<double(raw_state_bytes/(1ull<<40))
             <<" peer_gib_per_transpose="<<double((long double)peer_bytes_per_transpose/(1ull<<30))
             <<" logical_offdiag_gib="<<double((long double)logical_offdiag_bytes/(1ull<<30))
             <<" diagonal_gib="<<double((long double)diagonal_bytes/(1ull<<30))
             <<" slot_padding_peer_mib="<<double((long double)(peer_bytes_per_transpose-logical_offdiag_bytes)/(1ull<<20))
             <<" standard_transposes="<<standard_transposes
             <<" snake_transposes="<<snake_transposes
             <<" standard_peer_tib="<<double(standard_peer/(1ull<<40))
             <<" snake_peer_tib="<<double(snake_peer/(1ull<<40))
             <<" peer_saving_pct="<<double(saving*100.0L)
             <<"\n";

    if constexpr(TARGET_W==28){
        if(standard_transposes!=55||snake_transposes!=28)return 2;
        if(saving<0.49L||saving>0.491L)return 3;
        if(layout.main_size+layout.block_size!=520735012027ULL)return 4;
    }
    return 0;
}
