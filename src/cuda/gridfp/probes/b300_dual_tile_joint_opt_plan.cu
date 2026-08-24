#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>

#define RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../oneesan_cuda_gridfp_ramstream32_factorized_bidesc_compact.cu"
#undef RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../ramstream32_b300_dual_tile_joint_opt.cuh"

static double dtj_gib(long double x){return double(x/(1ull<<30));}
static double dtj_tib(long double x){return double(x/(1ull<<40));}

int main(){
    constexpr int NG=8;
    build_full_dp();G_FACTOR=build_factor_tables();
    StorageFactorHost f=build_storage_factor_tables(G_FACTOR);StorageLayout l=build_storage_layout(f);
    B300DualJointOptStats s{};
    B300DualTileHost z=build_b300_dual_tile_layout_joint_optimized(f,l,NG,4.0,256,4,&s);

    uint64_t before=s.logical_bytes-s.retained_initial,after=s.logical_bytes-s.retained_final;
    long double exact=0,l2h=0,h2l=0,maxarena=0,minarena=1e100L;
    std::array<long double,MAXGPU> lsend{},lrecv{},hsend{},hrecv{};
    for(int hi=0;hi<NG;++hi)for(int lo=0;lo<NG;++lo)if(hi!=lo){
        long double m=(long double)z.pair_main_size[hi][lo]*sizeof(Count),b=(long double)z.pair_block_size[hi][lo]*sizeof(Count);
        l2h+=m+b;h2l+=m;lsend[lo]+=m+b;lrecv[hi]+=m+b;hsend[hi]+=m;hrecv[lo]+=m;
    }
    exact=(long double)TARGET_W*l2h+(long double)(TARGET_W-1)*h2l;
    if(uint64_t(exact)!=after){std::cerr<<"joint objective mismatch exact="<<uint64_t(exact)<<" model="<<after<<'\n';return 570;}
    long double maxport=0;
    for(int g=0;g<NG;++g){long double arena=(long double)(z.main_count[g]+z.block_count[g])*sizeof(Count);maxarena=std::max(maxarena,arena);minarena=std::min(minarena,arena);
        maxport=std::max(maxport,(long double)TARGET_W*std::max(lsend[g],lrecv[g])+(long double)(TARGET_W-1)*std::max(hsend[g],hrecv[g]));}

    std::cout<<std::fixed<<std::setprecision(6)
        <<"b300-dual-tile-joint-opt-plan W="<<TARGET_W<<" gpus="<<NG
        <<" iterations="<<s.iterations<<" moves="<<s.moves<<" swaps="<<s.swaps
        <<" high_load_max_over_avg="<<s.high_load_max_over_avg
        <<" low_load_max_over_avg="<<s.low_load_max_over_avg
        <<" initial_offgpu_tib_per_residue="<<dtj_tib(before)
        <<" joint_offgpu_tib_per_residue="<<dtj_tib(after)
        <<" transfer_reduction="<<(before?1.0-double(after)/double(before):0.0)
        <<" retained_fraction="<<(s.logical_bytes?double(s.retained_final)/double(s.logical_bytes):0.0)
        <<" low_to_high_gib_per_row="<<dtj_gib(l2h)
        <<" high_to_low_main_gib_per_row="<<dtj_gib(h2l)
        <<" pairslot_arena_min_gib="<<dtj_gib(minarena)
        <<" pairslot_arena_max_gib="<<dtj_gib(maxarena)
        <<" max_gpu_port_tib_per_residue="<<dtj_tib(maxport)
        <<" ideal_1p8TBs_port_seconds_per_residue="<<double(maxport/1.8e12L)
        <<'\n';
    return 0;
}
