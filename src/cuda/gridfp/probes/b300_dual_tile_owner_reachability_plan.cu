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

constexpr int NG=8;
using TileMask=uint64_t; // bit hi*8+lo
struct Reach { std::array<TileMask,64> main{}; std::array<TileMask,32> block{}; };
static inline uint64_t bit(int hi,int lo){return 1ull<<(hi*NG+lo);}
static inline bool get(uint64_t m,int hi,int lo){return (m&bit(hi,lo))!=0;}
static inline void put(uint64_t& m,int hi,int lo){m|=bit(hi,lo);}

static int low_owner(const B300DualTileHost&z,const StorageFactorHost&f,
                     const StorageBlock&b,uint32_t lr){
    return z.low_owner[f.low_all_off[b.hs]+lr];
}
static int high_owner(const B300DualTileHost&z,const StorageFactorHost&f,
                      const StorageBlock&b,uint32_t hr){
    return z.high.high_owner[f.high_all_off[b.he]+hr];
}

uint32_t locate_main_block(Code rank,const StorageLayout&l,uint32_t&hr,uint32_t&lr){
    for(uint32_t b=0;b<l.main_blocks.size();++b){const auto&x=l.main_blocks[b];if(!x.valid)continue;
        Code n=Code(x.rows)*x.cols;if(rank>=x.off&&rank<x.off+n){Code q=rank-x.off;hr=uint32_t(q/x.cols);lr=uint32_t(q%x.cols);return b;}}
    std::exit(640);
}

void high_edge(Reach&r,const B300SparseActionsHost&s,const StorageFactorHost&f,
               const StorageLayout&l,const B300DualTileHost&z,int p){
    uint32_t pi=uint32_t((TARGET_W-1)-p);
    for(uint32_t q=s.high_orbit_off[pi];q<s.high_orbit_off[pi+1];++q){
        const auto&op=s.high_orbit[q];
        uint32_t sb=b300_sparse_sblock(op),jb=b300_sparse_jblock(op),db=b300_sparse_dblock(op);
        const auto&sx=l.main_blocks[sb];const auto&jx=l.main_blocks[jb];const auto&dx=l.block_blocks[db];
        int so=high_owner(z,f,sx,b300_sparse_src(op));
        int jo=high_owner(z,f,jx,b300_sparse_jrank(op));
        int bo=high_owner(z,f,dx,b300_sparse_drank(op));
        uint32_t kind=b300_sparse_kind(op);
        for(int lo=0;lo<NG;++lo){
            bool sm=get(r.main[sb],so,lo),jm=get(r.main[jb],jo,lo),bd=get(r.block[db],bo,lo);
            if(kind==HIGH_ORBIT_NN){
                if(jm||sm)put(r.main[jb],jo,lo);
                if(sm||bd)put(r.main[sb],so,lo);
                // Rank-level dp is cleared, but the owner tile can contain other
                // blocked ranks, so never clear a coarse tile here.
            }else{
                if(sm||jm||bd)put(r.main[sb],so,lo);
                if(bd||sm)put(r.block[db],bo,lo);
            }
        }
    }
    for(uint32_t q=s.high_closure_off[pi];q<s.high_closure_off[pi+1];++q){
        uint64_t op=s.high_closure[q];uint32_t sb=b300_sparse_closure_sblock(op),src=b300_sparse_closure_src(op);
        uint32_t d=b300_sparse_closure_desc(op),kind=b300_host_high_kind(d);
        if(kind!=HIGHDESC_BLOCK&&kind!=HIGHDESC_CROSS)continue;
        uint32_t db=(d>>HIGHDESC_BLOCK_SHIFT)&HIGHDESC_BLOCK_MASK;
        const auto&sx=l.main_blocks[sb];const auto&dx=l.block_blocks[db];
        int so=high_owner(z,f,sx,src),bo=high_owner(z,f,dx,d&HIGHDESC_RANK_MASK);
        for(int lo=0;lo<NG;++lo)if(get(r.main[sb],so,lo))
            put(r.block[db],bo,lo); // HIGH CROSS preserves LOW occupancy/owner.
    }
}

void low_edge(Reach&r,const B300SparseActionsHost&s,const StorageFactorHost&f,
              const StorageLayout&l,const B300DualTileHost&z,int p){
    uint32_t pi=uint32_t(LOW_LUT_K-p);
    for(uint32_t q=s.low_orbit_off[pi];q<s.low_orbit_off[pi+1];++q){
        const auto&op=s.low_orbit[q];
        uint32_t sb=b300_sparse_sblock(op),jb=b300_sparse_jblock(op),db=b300_sparse_dblock(op);
        const auto&sx=l.main_blocks[sb];const auto&jx=l.main_blocks[jb];const auto&dx=l.block_blocks[db];
        int so=low_owner(z,f,sx,b300_sparse_src(op));
        int jo=low_owner(z,f,jx,b300_sparse_jrank(op));
        int bo=low_owner(z,f,dx,b300_sparse_drank(op));
        uint32_t kind=b300_sparse_kind(op);
        for(int hi=0;hi<NG;++hi){
            bool sm=get(r.main[sb],hi,so),jm=get(r.main[jb],hi,jo),bd=get(r.block[db],hi,bo);
            if(kind==CPU_ORBIT_NN){
                if(jm||sm)put(r.main[jb],hi,jo);
                if(sm||bd)put(r.main[sb],hi,so);
            }else if(p==1){
                if(sm||jm||bd)put(r.main[sb],hi,so);
                if(jm||sm)put(r.main[jb],hi,jo);
            }else{
                if(sm||jm||bd)put(r.main[sb],hi,so);
                if(bd||sm)put(r.block[db],hi,bo);
            }
        }
    }
    for(uint32_t q=s.low_closure_off[pi];q<s.low_closure_off[pi+1];++q){
        uint64_t op=s.low_closure[q];uint32_t sb=b300_sparse_closure_sblock(op),src=b300_sparse_closure_src(op);
        uint32_t d=b300_sparse_closure_desc(op),kind=b300_host_low_kind(d);
        uint32_t db=(d>>LOWDESC_BLOCK_SHIFT)&LOWDESC_BLOCK_MASK;
        const auto&sx=l.main_blocks[sb];
        int so=low_owner(z,f,sx,src),doo=-1;
        bool main_dest=false;
        if(kind==LOWDESC_MAIN){main_dest=true;doo=low_owner(z,f,l.main_blocks[db],d&LOWDESC_LR_MASK);}
        else if(kind==LOWDESC_BLOCK){doo=low_owner(z,f,l.block_blocks[db],d&LOWDESC_LR_MASK);}
        else if(kind==LOWDESC_CROSS){
            main_dest=(p==1);
            if(main_dest)doo=low_owner(z,f,l.main_blocks[db],d&LOWDESC_LR_MASK);
            else doo=low_owner(z,f,l.block_blocks[db],d&LOWDESC_LR_MASK);
        }else continue;
        for(int hi=0;hi<NG;++hi)if(get(r.main[sb],hi,so)){
            if(main_dest)put(r.main[db],hi,doo);else put(r.block[db],hi,doo);
            // LOW CROSS preserves HIGH occupancy/owner, hence the same hi.
        }
    }
    if(p==1)r.block.fill(0); // exact row-boundary invariant
}

long double active_bytes(const Reach&r,const B300DualTileHost&z,const StorageLayout&l,
                         bool mainv,bool blockv){
    long double out=0;
    if(mainv)for(uint32_t bid=0;bid<l.main_blocks.size();++bid){const auto&b=l.main_blocks[bid];if(!b.valid)continue;
        for(int hi=0;hi<NG;++hi)for(int lo=0;lo<NG;++lo)if(hi!=lo&&get(r.main[bid],hi,lo))
            out+=(long double)z.high_count[hi][b.he]*z.low_count[lo][b.hs]*sizeof(Count);}
    if(blockv)for(uint32_t bid=0;bid<l.block_blocks.size();++bid){const auto&b=l.block_blocks[bid];if(!b.valid)continue;
        for(int hi=0;hi<NG;++hi)for(int lo=0;lo<NG;++lo)if(hi!=lo&&get(r.block[bid],hi,lo))
            out+=(long double)z.high_count[hi][b.he]*z.low_count[lo][b.hs]*sizeof(Count);}
    return out;
}
int tiles(const std::array<TileMask,64>&a){int n=0;for(auto x:a)n+=__builtin_popcountll(x);return n;}
int tilesb(const std::array<TileMask,32>&a){int n=0;for(auto x:a)n+=__builtin_popcountll(x);return n;}
static double gib(long double x){return double(x/(1ull<<30));}
static double tib(long double x){return double(x/(1ull<<40));}

} // namespace

int main(){
    build_full_dp();G_FACTOR=build_factor_tables();
    StorageFactorHost f=build_storage_factor_tables(G_FACTOR);StorageLayout l=build_storage_layout(f);
    LowDescHost ld=build_low_descriptors(f,l);HighDescHost hd=build_high_descriptors(f,l);
    LowOrbitHost lo=build_cpu_low_orbit(f,l,ld);HighOrbitHost ho=build_high_orbit(f,l);
    B300SparseActionsHost s=build_b300_sparse_actions(l,ld,lo,hd,ho);
    B300DualTileHost z=build_b300_dual_tile_layout_w28_precomputed(f,l,NG);

    Reach r;uint32_t hr=0,lr=0;
    Code rank=storage_rank_main_host(MateID(R)<<(2*(TARGET_W-1)),f,l);
    uint32_t ib=locate_main_block(rank,l,hr,lr);
    int hi=high_owner(z,f,l.main_blocks[ib],hr),loo=low_owner(z,f,l.main_blocks[ib],lr);
    put(r.main[ib],hi,loo);

    Reach all;for(uint32_t b=0;b<l.main_blocks.size();++b)if(l.main_blocks[b].valid)all.main[b]=~0ull;
    for(uint32_t b=0;b<l.block_blocks.size();++b)if(l.block_blocks[b].valid)all.block[b]=~0ull;
    long double full_l2h=active_bytes(all,z,l,true,true),full_h2l=active_bytes(all,z,l,true,false);
    long double active_total=0,full_total=0;
    std::cout<<std::fixed<<std::setprecision(6)
        <<"b300-dual-tile-owner-reachability W="<<TARGET_W
        <<" init_block="<<ib<<" init_hi="<<hi<<" init_lo="<<loo
        <<" full_l2h_gib="<<gib(full_l2h)<<" full_h2l_gib="<<gib(full_h2l)<<'\n';

    for(int row=0;row<TARGET_W;++row){
        for(int p=TARGET_W-1;p>=LOW_LUT_K+1;--p)high_edge(r,s,f,l,z,p);
        long double a=active_bytes(r,z,l,true,true);active_total+=a;full_total+=full_l2h;
        std::cout<<"owner_reach_row="<<row<<" stage=L2H main_tiles="<<tiles(r.main)
                 <<" block_tiles="<<tilesb(r.block)<<" active_gib="<<gib(a)
                 <<" full_fraction="<<double(a/full_l2h)<<'\n';
        for(int p=LOW_LUT_K;p>=1;--p)low_edge(r,s,f,l,z,p);
        if(row+1<TARGET_W){a=active_bytes(r,z,l,true,false);active_total+=a;full_total+=full_h2l;
            std::cout<<"owner_reach_row="<<row<<" stage=H2L main_tiles="<<tiles(r.main)
                     <<" block_tiles="<<tilesb(r.block)<<" active_gib="<<gib(a)
                     <<" full_fraction="<<double(a/full_h2l)<<'\n';}
    }
    std::cout<<"owner_reachability_summary full_tib_per_residue="<<tib(full_total)
             <<" conservative_pruned_tib_per_residue="<<tib(active_total)
             <<" reduction="<<(full_total?1.0-double(active_total/full_total):0.0)
             <<" final_main_tiles="<<tiles(r.main)<<" final_block_tiles="<<tilesb(r.block)<<'\n';
    return 0;
}
