#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>

#define RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../oneesan_cuda_gridfp_ramstream32_factorized_bidesc_compact.cu"
#undef RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../ramstream32_b300_dual_tile_precomputed_w28.cuh"

static double dtp_gib(long double x){return double(x/(1ull<<30));}
static double dtp_tib(long double x){return double(x/(1ull<<40));}

int main(){
    constexpr int NG=8;
    build_full_dp();G_FACTOR=build_factor_tables();
    StorageFactorHost f=build_storage_factor_tables(G_FACTOR);StorageLayout l=build_storage_layout(f);
    B300DualTileHost z=build_b300_dual_tile_layout_w28_precomputed(f,l,NG);

    long double l2h=0,h2l=0,maxarena=0,minarena=1e100L;
    long double max_halfduplex_port=0,max_bidir_port=0;
    std::array<long double,MAXGPU>ls{},lr{},hs{},hr{};
    for(int hi=0;hi<NG;++hi)for(int lo=0;lo<NG;++lo)if(hi!=lo){
        long double m=(long double)z.pair_main_size[hi][lo]*sizeof(Count);
        long double b=(long double)z.pair_block_size[hi][lo]*sizeof(Count);
        l2h+=m+b;h2l+=m;ls[lo]+=m+b;lr[hi]+=m+b;hs[hi]+=m;hr[lo]+=m;
    }
    long double residue=(long double)TARGET_W*l2h+(long double)(TARGET_W-1)*h2l;
    for(int g=0;g<NG;++g){
        long double arena=(long double)(z.main_count[g]+z.block_count[g])*sizeof(Count);
        minarena=std::min(minarena,arena);maxarena=std::max(maxarena,arena);
        // Historical metric: assumes the quoted link number independently applies
        // to each direction. Keep it only as a balance regression check.
        long double half=(long double)TARGET_W*std::max(ls[g],lr[g])
            +(long double)(TARGET_W-1)*std::max(hs[g],hr[g]);
        max_halfduplex_port=std::max(max_halfduplex_port,half);
        // NVLink 5's quoted 1.8 TB/s is bidirectional aggregate throughput, so the
        // physically relevant lower bound charges both transmitted and received bytes.
        long double bidir=(long double)TARGET_W*(ls[g]+lr[g])
            +(long double)(TARGET_W-1)*(hs[g]+hr[g]);
        max_bidir_port=std::max(max_bidir_port,bidir);
    }

    // These are exact integer byte counts reconstructed independently from the
    // occupancy-popcount MILP quotas. Keep the tolerance below one MiB so any
    // ownership/order mismatch trips immediately.
    const long double want_residue=78.281909373L*(1ull<<40);
    const long double want_halfduplex_port=9.799945773L*(1ull<<40);
    const long double tol=1.0L*(1ull<<20);
    if(std::abs(residue-want_residue)>tol){
        std::cerr<<"precomputed residue traffic mismatch got="<<dtp_tib(residue)
                 <<" want=78.281909373\n";return 590;
    }
    if(std::abs(max_halfduplex_port-want_halfduplex_port)>tol){
        std::cerr<<"precomputed balance metric mismatch got="<<dtp_tib(max_halfduplex_port)
                 <<" want=9.799945773\n";return 591;
    }

    std::cout<<std::fixed<<std::setprecision(9)
        <<"b300-dual-tile-precomputed-plan OK W="<<TARGET_W<<" gpus="<<NG
        <<" low_to_high_gib_per_row="<<dtp_gib(l2h)
        <<" high_to_low_main_gib_per_row="<<dtp_gib(h2l)
        <<" offgpu_tib_per_residue="<<dtp_tib(residue)
        <<" legacy_one_direction_balance_tib="<<dtp_tib(max_halfduplex_port)
        <<" max_gpu_bidirectional_port_tib_per_residue="<<dtp_tib(max_bidir_port)
        <<" ideal_1p8TBs_bidirectional_seconds_per_residue="<<double(max_bidir_port/1.8e12L)
        <<" pairslot_arena_min_gib="<<dtp_gib(minarena)
        <<" pairslot_arena_max_gib="<<dtp_gib(maxarena)<<'\n';
    return 0;
}
