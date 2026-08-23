#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <iomanip>
#include <iostream>

#define RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../oneesan_cuda_gridfp_ramstream32_factorized_bidesc_compact.cu"
#undef RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../ramstream32_high_orbit.cuh"
#include "../ramstream32_cpu_low_inplace.hpp"
#include "../ramstream32_b300_dual_tile_pruned_plan.cuh"

static double tib(uint64_t x){return double((long double)x/(1ull<<40));}

int main(){
    constexpr int NG=8;
    build_full_dp();G_FACTOR=build_factor_tables();
    StorageFactorHost f=build_storage_factor_tables(G_FACTOR);StorageLayout l=build_storage_layout(f);
    LowDescHost ld=build_low_descriptors(f,l);HighDescHost hd=build_high_descriptors(f,l);
    LowOrbitHost lo=build_cpu_low_orbit(f,l,ld);HighOrbitHost ho=build_high_orbit(f,l);
    B300SparseActionsHost sparse=build_b300_sparse_actions(l,ld,lo,hd,ho);
    B300DualTileHost dual=build_b300_dual_tile_layout_w28_precomputed(f,l,NG);
    B300DualReachSchedule reach=build_b300_dual_reach_schedule(sparse,f,l,dual);

    const uint64_t mib=1ull<<20;
    const uint64_t penalties[]={0,1*mib,4*mib,8*mib,16*mib,64*mib,256*mib};
    uint64_t logical_ref=0,full_ref=0;
    std::cout<<std::fixed<<std::setprecision(9);
    for(uint64_t penalty:penalties){
        auto p=build_b300_dual_pruned_schedule_plan(dual,l,reach,penalty);
        uint64_t logical=p.logical_bytes_per_residue();
        uint64_t scheduled=p.scheduled_bytes_per_residue();
        uint64_t full=p.full_bytes_per_residue();
        if(!logical_ref){logical_ref=logical;full_ref=full;}
        if(logical!=logical_ref||full!=full_ref){std::cerr<<"pruned plan invariant changed with penalty\n";return 670;}
        auto ports=p.gpu_port_bytes_per_residue();
        uint64_t port_sum=0;for(int g=0;g<NG;++g)port_sum+=ports[g];
        if(port_sum!=2*scheduled){
            std::cerr<<"peer port accounting mismatch scheduled="<<scheduled
                     <<" port_sum="<<port_sum<<'\n';return 671;
        }
        uint64_t max_port=p.max_gpu_port_bytes_per_residue();
        long double avg_port=(2.0L*scheduled)/NG;
        std::cout<<"pruned_pareto penalty_mib="<<double(penalty)/double(mib)
                 <<" logical_tib="<<tib(logical)
                 <<" scheduled_tib="<<tib(scheduled)
                 <<" full_tib="<<tib(full)
                 <<" launches="<<p.launches_per_residue()
                 <<" max_gpu_port_tib="<<tib(max_port)
                 <<" port_max_over_avg="<<(avg_port?double(max_port/avg_port):0.0)
                 <<" ideal_1p8TBs_bidirectional_seconds="<<double((long double)max_port/1.8e12L)
                 <<" logical_reduction="<<(full?1.0-double(logical)/double(full):0.0)
                 <<" scheduled_reduction="<<(full?1.0-double(scheduled)/double(full):0.0)
                 <<'\n';
        for(int g=0;g<NG;++g)
            std::cout<<"pruned_port penalty_mib="<<double(penalty)/double(mib)
                     <<" gpu="<<g<<" tib="<<tib(ports[g])<<'\n';
    }
    return 0;
}
