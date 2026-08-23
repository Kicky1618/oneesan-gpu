#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iomanip>
#include <iostream>

#define RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../oneesan_cuda_gridfp_ramstream32_factorized_bidesc_compact.cu"
#undef RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../ramstream32_high_orbit.cuh"
#include "../ramstream32_cpu_low_inplace.hpp"
#include "../ramstream32_b300_sparse_actions.cuh"
#include "../ramstream32_b300_dual_tile_precomputed_w28.cuh"

namespace {

static int low_owner(const B300DualTileHost& z,const StorageFactorHost& f,
                     const StorageBlock& b,uint32_t lr){
    return z.low_owner[f.low_all_off[b.hs]+lr];
}
static int high_owner(const B300DualTileHost& z,const StorageFactorHost& f,
                      const StorageBlock& b,uint32_t hr){
    return z.high.high_owner[f.high_all_off[b.he]+hr];
}

// Three read-modify-write variables in every orbit operation. Choose the
// execution GPU among their owners to minimize remote variables. One remote
// uint32 variable costs a peer read + peer write = 8 bytes per opposite-axis
// element. This is a byte-traffic model, not a latency model.
static int remote_vars3(int a,int b,int c){
    int best=3;
    const int cand[3]={a,b,c};
    for(int q=0;q<3;++q){int g=cand[q],r=(a!=g)+(b!=g)+(c!=g);best=std::min(best,r);}
    return best;
}

long double main_block_offgpu(const B300DualTileHost& z,const StorageLayout& l,uint32_t bid){
    const auto& b=l.main_blocks[bid];if(!b.valid)return 0;
    long double n=0;for(int hi=0;hi<z.ngpu;++hi)for(int lo=0;lo<z.ngpu;++lo)if(hi!=lo)
        n+=(long double)z.high_count[hi][b.he]*z.low_count[lo][b.hs]*sizeof(Count);
    return n;
}
long double block_block_offgpu(const B300DualTileHost& z,const StorageLayout& l,uint32_t bid){
    const auto& b=l.block_blocks[bid];if(!b.valid)return 0;
    long double n=0;for(int hi=0;hi<z.ngpu;++hi)for(int lo=0;lo<z.ngpu;++lo)if(hi!=lo)
        n+=(long double)z.high_count[hi][b.he]*z.low_count[lo][b.hs]*sizeof(Count);
    return n;
}

static double gib(long double x){return double(x/(1ull<<30));}
static double tib(long double x){return double(x/(1ull<<40));}

} // namespace

int main(){
    constexpr int NG=8;
    build_full_dp();G_FACTOR=build_factor_tables();
    StorageFactorHost f=build_storage_factor_tables(G_FACTOR);StorageLayout l=build_storage_layout(f);
    LowDescHost ld=build_low_descriptors(f,l);HighDescHost hd=build_high_descriptors(f,l);
    LowOrbitHost lo=build_cpu_low_orbit(f,l,ld);HighOrbitHost ho=build_high_orbit(f,l);
    B300SparseActionsHost s=build_b300_sparse_actions(l,ld,lo,hd,ho);
    B300DualTileHost z=build_b300_dual_tile_layout_w28_precomputed(f,l,NG);

    std::array<long double,64> low_wrong_main{};  // LOW window while still LOW-owned
    std::array<long double,64> high_wrong_main{}; // HIGH window while still HIGH-owned

    // LOW orbit: owners are determined by LOW all-ranks while every operation
    // spans all HIGH rows. Attribute the operation to its source main block.
    for(const auto&op:s.low_orbit){
        uint32_t sb=b300_sparse_sblock(op),jb=b300_sparse_jblock(op),db=b300_sparse_dblock(op);
        const auto&x=l.main_blocks[sb];const auto&y=l.main_blocks[jb];const auto&d=l.block_blocks[db];
        int a=low_owner(z,f,x,b300_sparse_src(op));
        int b=low_owner(z,f,y,b300_sparse_jrank(op));
        int c=low_owner(z,f,d,b300_sparse_drank(op));
        low_wrong_main[sb]+=(long double)remote_vars3(a,b,c)*8.0L*x.rows;
    }
    // LOW closure: execute on destination owner. CROSS changes target storage
    // class at p=1: p=1 targets MAIN, while p>1 targets BLOCKED. Therefore the
    // flat stream must be decoded edge-by-edge rather than as one undifferentiated list.
    for(int p=LOW_LUT_K;p>=1;--p){
        uint32_t pi=uint32_t(LOW_LUT_K-p);
        for(uint32_t q=s.low_closure_off[pi];q<s.low_closure_off[pi+1];++q){
            uint64_t op=s.low_closure[q];
            uint32_t sb=b300_sparse_closure_sblock(op),src=b300_sparse_closure_src(op);
            uint32_t desc=b300_sparse_closure_desc(op),kind=b300_host_low_kind(desc);
            uint32_t db=(desc>>LOWDESC_BLOCK_SHIFT)&LOWDESC_BLOCK_MASK;
            const auto&x=l.main_blocks[sb];
            const StorageBlock* y=nullptr;
            if(kind==LOWDESC_MAIN)y=&l.main_blocks[db];
            else if(kind==LOWDESC_BLOCK)y=&l.block_blocks[db];
            else if(kind==LOWDESC_CROSS)y=(p==1?&l.main_blocks[db]:&l.block_blocks[db]);
            if(!y)continue;
            int a=low_owner(z,f,x,src);
            int b=low_owner(z,f,*y,desc&LOWDESC_LR_MASK);
            if(a!=b)low_wrong_main[sb]+=(long double)4*x.rows;
        }
    }

    // HIGH orbit: symmetric model in HIGH all-ranks, spanning LOW columns.
    for(const auto&op:s.high_orbit){
        uint32_t sb=b300_sparse_sblock(op),jb=b300_sparse_jblock(op),db=b300_sparse_dblock(op);
        const auto&x=l.main_blocks[sb];const auto&y=l.main_blocks[jb];const auto&d=l.block_blocks[db];
        int a=high_owner(z,f,x,b300_sparse_src(op));
        int b=high_owner(z,f,y,b300_sparse_jrank(op));
        int c=high_owner(z,f,d,b300_sparse_drank(op));
        high_wrong_main[sb]+=(long double)remote_vars3(a,b,c)*8.0L*x.cols;
    }
    for(uint64_t op:s.high_closure){
        uint32_t sb=b300_sparse_closure_sblock(op),src=b300_sparse_closure_src(op);
        uint32_t desc=b300_sparse_closure_desc(op),kind=b300_host_high_kind(desc);
        if(kind!=HIGHDESC_BLOCK&&kind!=HIGHDESC_CROSS)continue;
        uint32_t db=(desc>>HIGHDESC_BLOCK_SHIFT)&HIGHDESC_BLOCK_MASK;
        const auto&x=l.main_blocks[sb];const auto&y=l.block_blocks[db];
        int a=high_owner(z,f,x,src);
        int b=high_owner(z,f,y,desc&HIGHDESC_RANK_MASK);
        if(a!=b)high_wrong_main[sb]+=(long double)4*x.cols;
    }

    // A MAIN block kept permanently LOW saves all of its L2H and H2L moves and
    // pays wrong-orientation peer traffic only during the LOW window. Symmetric
    // for a permanently HIGH block. These ratios are only a screen: actions
    // couple source/partner/destination blocks, so candidates must be solved as
    // a joint mixed-orientation problem before runtime use.
    long double low_candidate_saved=0,low_candidate_peer=0;
    long double high_candidate_saved=0,high_candidate_peer=0;
    int low_candidates=0,high_candidates=0;
    for(uint32_t bid=0;bid<l.main_blocks.size();++bid){
        if(!l.main_blocks[bid].valid)continue;
        long double one=main_block_offgpu(z,l,bid);
        long double saved=(long double)(2*TARGET_W-1)*one;
        long double lowpeer=(long double)TARGET_W*low_wrong_main[bid];
        long double highpeer=(long double)TARGET_W*high_wrong_main[bid];
        if(lowpeer<saved){++low_candidates;low_candidate_saved+=saved;low_candidate_peer+=lowpeer;}
        if(highpeer<saved){++high_candidates;high_candidate_saved+=saved;high_candidate_peer+=highpeer;}
        std::cout<<std::fixed<<std::setprecision(6)
            <<"hybrid_main_block="<<bid
            <<" transpose_gib_per_residue="<<gib(saved)
            <<" low_wrong_peer_gib="<<gib(lowpeer)
            <<" low_ratio="<<(saved?double(lowpeer/saved):0.0)
            <<" high_wrong_peer_gib="<<gib(highpeer)
            <<" high_ratio="<<(saved?double(highpeer/saved):0.0)<<'\n';
    }

    long double block_shuffle=0;
    for(uint32_t bid=0;bid<l.block_blocks.size();++bid)if(l.block_blocks[bid].valid)
        block_shuffle+=(long double)TARGET_W*block_block_offgpu(z,l,bid);

    std::cout<<std::fixed<<std::setprecision(6)
        <<"hybrid_screen_summary"
        <<" low_permanent_candidates="<<low_candidates
        <<" low_candidate_transpose_tib="<<tib(low_candidate_saved)
        <<" low_candidate_peer_tib="<<tib(low_candidate_peer)
        <<" high_permanent_candidates="<<high_candidates
        <<" high_candidate_transpose_tib="<<tib(high_candidate_saved)
        <<" high_candidate_peer_tib="<<tib(high_candidate_peer)
        <<" blocked_total_shuffle_tib="<<tib(block_shuffle)
        <<" note=mixed_orientation_coupling_not_yet_modeled\n";
    return 0;
}
