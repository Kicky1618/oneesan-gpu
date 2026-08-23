#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>

#define RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../oneesan_cuda_gridfp_ramstream32_factorized_bidesc_compact.cu"
#undef RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../ramstream32_b300_dual_tile_opt.cuh"

static double dtop_gib(long double x){return double(x/(1ull<<30));}
static double dtop_tib(long double x){return double(x/(1ull<<40));}

int main(){
    constexpr int NG=8;
    build_full_dp();G_FACTOR=build_factor_tables();
    StorageFactorHost f=build_storage_factor_tables(G_FACTOR);StorageLayout l=build_storage_layout(f);
    B300DualTileHost z=build_b300_dual_tile_layout(f,l,NG);

    B300DualLowOptStats s{};
    std::vector<uint8_t> owner=b300_dt_optimize_low_mask_owner(f,l,z.high,NG,4.0,512,&s);
    uint64_t before=s.logical_bytes-s.retained_before;
    uint64_t after=s.logical_bytes-s.retained_after;
    b300_dt_rebuild_low_owner(z,owner,f,l);

    long double maxarena=0,minarena=1e100L;
    for(int g=0;g<NG;++g){long double x=(long double)(z.main_count[g]+z.block_count[g])*sizeof(Count);maxarena=std::max(maxarena,x);minarena=std::min(minarena,x);}

    long double l2h=0,h2l=0;
    for(int hi=0;hi<NG;++hi)for(int lo=0;lo<NG;++lo)if(hi!=lo){
        l2h+=(long double)(z.pair_main_size[hi][lo]+z.pair_block_size[hi][lo])*sizeof(Count);
        h2l+=(long double)z.pair_main_size[hi][lo]*sizeof(Count);
    }
    long double exact=(long double)TARGET_W*l2h+(long double)(TARGET_W-1)*h2l;
    if(uint64_t(exact)!=after){
        std::cerr<<"dual optimized objective mismatch exact="<<uint64_t(exact)<<" model="<<after<<'\n';return 560;
    }

    std::cout<<std::fixed<<std::setprecision(6)
        <<"b300-dual-tile-opt-plan W="<<TARGET_W<<" gpus="<<NG
        <<" moves="<<s.moves<<" swaps="<<s.swaps
        <<" low_load_max_over_avg="<<s.load_max_over_avg
        <<" lpt_offgpu_tib_per_residue="<<dtop_tib(before)
        <<" optimized_offgpu_tib_per_residue="<<dtop_tib(after)
        <<" transfer_reduction="<<(before?1.0-double(after)/double(before):0.0)
        <<" retained_fraction="<<(s.logical_bytes?double(s.retained_after)/double(s.logical_bytes):0.0)
        <<" pairslot_arena_min_gib="<<dtop_gib(minarena)
        <<" pairslot_arena_max_gib="<<dtop_gib(maxarena)
        <<" low_to_high_offgpu_gib_per_row="<<dtop_gib(l2h)
        <<" high_to_low_main_offgpu_gib_per_row="<<dtop_gib(h2l)
        <<'\n';
    return 0;
}
