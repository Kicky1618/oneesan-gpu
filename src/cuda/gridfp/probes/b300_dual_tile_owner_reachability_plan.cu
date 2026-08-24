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
#include "../ramstream32_b300_dual_tile_reachability.cuh"

namespace {
constexpr int NG=8;

long double active_bytes(
    const B300DualReachStage&r,const B300DualTileHost&z,const StorageLayout&l,
    bool mainv,bool blockv
){
    long double out=0;
    if(mainv)for(uint32_t bid=0;bid<l.main_blocks.size();++bid){
        const auto&b=l.main_blocks[bid];if(!b.valid)continue;
        for(int hi=0;hi<z.ngpu;++hi)for(int lo=0;lo<z.ngpu;++lo)
            if(hi!=lo&&b300_dt_reach_main_active(r,bid,hi,lo,z.ngpu))
                out+=(long double)z.high_count[hi][b.he]*z.low_count[lo][b.hs]*sizeof(Count);
    }
    if(blockv)for(uint32_t bid=0;bid<l.block_blocks.size();++bid){
        const auto&b=l.block_blocks[bid];if(!b.valid)continue;
        for(int hi=0;hi<z.ngpu;++hi)for(int lo=0;lo<z.ngpu;++lo)
            if(hi!=lo&&b300_dt_reach_block_active(r,bid,hi,lo,z.ngpu))
                out+=(long double)z.high_count[hi][b.he]*z.low_count[lo][b.hs]*sizeof(Count);
    }
    return out;
}
int main_tiles(const B300DualReachStage&r){int n=0;for(auto x:r.main)n+=__builtin_popcountll(x);return n;}
int block_tiles(const B300DualReachStage&r){int n=0;for(auto x:r.block)n+=__builtin_popcountll(x);return n;}
static double gib(long double x){return double(x/(1ull<<30));}
static double tib(long double x){return double(x/(1ull<<40));}
}

int main(){
    build_full_dp();G_FACTOR=build_factor_tables();
    StorageFactorHost f=build_storage_factor_tables(G_FACTOR);StorageLayout l=build_storage_layout(f);
    LowDescHost ld=build_low_descriptors(f,l);HighDescHost hd=build_high_descriptors(f,l);
    LowOrbitHost lo=build_cpu_low_orbit(f,l,ld);HighOrbitHost ho=build_high_orbit(f,l);
    B300SparseActionsHost s=build_b300_sparse_actions(l,ld,lo,hd,ho);
    B300DualTileHost z=build_b300_dual_tile_layout_w28_precomputed(f,l,NG);
    B300DualReachSchedule reach=build_b300_dual_reach_schedule(s,f,l,z);
    if(reach.l2h.size()!=TARGET_W||reach.h2l.size()!=TARGET_W-1)return 640;

    B300DualReachStage all;
    for(uint32_t b=0;b<l.main_blocks.size();++b)if(l.main_blocks[b].valid)all.main[b]=~0ull;
    for(uint32_t b=0;b<l.block_blocks.size();++b)if(l.block_blocks[b].valid)all.block[b]=~0ull;
    long double full_l2h=active_bytes(all,z,l,true,true);
    long double full_h2l=active_bytes(all,z,l,true,false);
    long double active_total=0,full_total=0;

    std::cout<<std::fixed<<std::setprecision(6)
        <<"b300-dual-tile-owner-reachability W="<<TARGET_W
        <<" full_l2h_gib="<<gib(full_l2h)
        <<" full_h2l_gib="<<gib(full_h2l)<<'\n';

    for(int row=0;row<TARGET_W;++row){
        const auto&a=reach.l2h[row];
        long double bytes=active_bytes(a,z,l,true,true);
        active_total+=bytes;full_total+=full_l2h;
        std::cout<<"owner_reach_row="<<row<<" stage=L2H main_tiles="<<main_tiles(a)
                 <<" block_tiles="<<block_tiles(a)<<" active_gib="<<gib(bytes)
                 <<" full_fraction="<<double(bytes/full_l2h)<<'\n';
        if(row+1<TARGET_W){
            const auto&b=reach.h2l[row];
            bytes=active_bytes(b,z,l,true,false);
            active_total+=bytes;full_total+=full_h2l;
            std::cout<<"owner_reach_row="<<row<<" stage=H2L main_tiles="<<main_tiles(b)
                     <<" block_tiles="<<block_tiles(b)<<" active_gib="<<gib(bytes)
                     <<" full_fraction="<<double(bytes/full_h2l)<<'\n';
        }
    }
    std::cout<<"owner_reachability_summary full_tib_per_residue="<<tib(full_total)
             <<" conservative_pruned_tib_per_residue="<<tib(active_total)
             <<" reduction="<<(full_total?1.0-double(active_total/full_total):0.0)
             <<" final_main_tiles="<<main_tiles(reach.l2h.back())
             <<" final_block_tiles="<<block_tiles(reach.l2h.back())<<'\n';
    return 0;
}
