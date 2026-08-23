#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <vector>

#define RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../oneesan_cuda_gridfp_ramstream32_factorized_bidesc_compact.cu"
#undef RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../ramstream32_high_orbit.cuh"
#include "../ramstream32_cpu_low_inplace.hpp"
#include "../ramstream32_b300_dual_tile_layout.cuh"
#include "../ramstream32_b300_sparse_actions.cuh"

static uint32_t dt_flip_low_host(uint32_t lc,uint32_t depth){
    int s=int(depth);
    for(int pos=LOW_LUT_K-1;pos>=0;--pos){
        MateValue v=MateValue((lc>>(2*pos))&3u);
        if(v==::L)++s;
        else if(v==R&&--s==0){uint32_t z=3u<<(2*pos);return(lc&~z)|(uint32_t(::L)<<(2*pos));}
    }
    return 0xffffffffu;
}
static uint32_t dt_flip_high_host(uint32_t hc,uint32_t depth){
    int s=int(depth);
    for(int pos=0;pos<HIGH_LUT_K;++pos){
        MateValue v=MateValue((hc>>(2*pos))&3u);
        if(v==::L){if(--s==0){uint32_t z=3u<<(2*pos);return(hc&~z)|(uint32_t(R)<<(2*pos));}}
        else if(v==R)++s;
    }
    return 0xffffffffu;
}

static Code dt_host_main_index(
    const B300DualTileHost& z,const StorageFactorHost& f,const StorageLayout& l,
    int bid,uint32_t hr,uint32_t lr,bool high,int& owner
){
    const auto&b=l.main_blocks[bid];
    uint32_t hai=f.high_all_off[b.he]+hr,lai=f.low_all_off[b.hs]+lr;
    int hi=z.high.high_owner[hai],lo=z.low_owner[lai];
    uint32_t h=z.high.high_local[hai],q=z.low_local[lai];
    size_t pi=(size_t(hi)*z.ngpu+lo)*l.main_blocks.size()+bid;
    Code k=z.pair_main_off[pi]+Code(h)*z.low_count[lo][b.hs]+q;
    owner=high?hi:lo;
    return z.main_slot_base[owner][high?lo:hi]+k;
}
static Code dt_host_block_index(
    const B300DualTileHost& z,const StorageFactorHost& f,const StorageLayout& l,
    int bid,uint32_t hr,uint32_t lr,bool high,int& owner
){
    const auto&b=l.block_blocks[bid];
    uint32_t hai=f.high_all_off[b.he]+hr,lai=f.low_all_off[b.hs]+lr;
    int hi=z.high.high_owner[hai],lo=z.low_owner[lai];
    uint32_t h=z.high.high_local[hai],q=z.low_local[lai];
    size_t pi=(size_t(hi)*z.ngpu+lo)*l.block_blocks.size()+bid;
    Code k=z.pair_block_off[pi]+Code(h)*z.low_count[lo][b.hs]+q;
    owner=high?hi:lo;
    return z.block_slot_base[owner][high?lo:hi]+k;
}

template<class T>
static void dt_cpu_swap(
    const B300DualTileHost&z,std::vector<std::vector<T>>&v,bool blocked,bool low_to_high
){
    const auto&sz=blocked?z.pair_block_size:z.pair_main_size;
    const auto&bs=blocked?z.block_slot_base:z.main_slot_base;
    for(int a=0;a<z.ngpu;++a)for(int b=a+1;b<z.ngpu;++b){
        Code na=low_to_high?sz[b][a]:sz[a][b];
        Code nb=low_to_high?sz[a][b]:sz[b][a];
        std::vector<T> ta(size_t(na)),tb(size_t(nb));
        for(Code q=0;q<na;++q)ta[size_t(q)]=v[a][size_t(bs[a][b]+q)];
        for(Code q=0;q<nb;++q)tb[size_t(q)]=v[b][size_t(bs[b][a]+q)];
        for(Code q=0;q<na;++q)v[b][size_t(bs[b][a]+q)]=ta[size_t(q)];
        for(Code q=0;q<nb;++q)v[a][size_t(bs[a][b]+q)]=tb[size_t(q)];
    }
}

int main(){
    constexpr int NG=4;
    build_full_dp();G_FACTOR=build_factor_tables();
    StorageFactorHost f=build_storage_factor_tables(G_FACTOR);StorageLayout l=build_storage_layout(f);
    LowDescHost ld=build_low_descriptors(f,l);HighDescHost hd=build_high_descriptors(f,l);
    LowOrbitHost lo=build_cpu_low_orbit(f,l,ld);HighOrbitHost ho=build_high_orbit(f,l);
    B300SparseActionsHost sparse=build_b300_sparse_actions(l,ld,lo,hd,ho);
    B300DualTileHost z=build_b300_dual_tile_layout(f,l,NG);

    std::vector<std::vector<uint64_t>> mv(NG),bv(NG);
    for(int g=0;g<NG;++g){mv[g].assign(size_t(z.main_count[g]),0);bv[g].assign(size_t(z.block_count[g]),0);}

    // Materialize the authoritative vectors directly in LOW orientation.
    for(int bid=0;bid<int(l.main_blocks.size());++bid){const auto&b=l.main_blocks[bid];if(!b.valid)continue;
        for(uint32_t hr=0;hr<b.rows;++hr)for(uint32_t lr=0;lr<b.cols;++lr){int g=0;Code k=dt_host_main_index(z,f,l,bid,hr,lr,false,g);
            if(k>=z.main_count[g])return 540;mv[g][size_t(k)]=uint64_t(b.off+Code(hr)*b.cols+lr)+1;}}
    for(int bid=0;bid<int(l.block_blocks.size());++bid){const auto&b=l.block_blocks[bid];if(!b.valid)continue;
        for(uint32_t hr=0;hr<b.rows;++hr)for(uint32_t lr=0;lr<b.cols;++lr){int g=0;Code k=dt_host_block_index(z,f,l,bid,hr,lr,false,g);
            if(k>=z.block_count[g])return 541;bv[g][size_t(k)]=uint64_t(b.off+Code(hr)*b.cols+lr)+1;}}

    dt_cpu_swap(z,mv,false,true);dt_cpu_swap(z,bv,true,true);
    for(int bid=0;bid<int(l.main_blocks.size());++bid){const auto&b=l.main_blocks[bid];if(!b.valid)continue;
        for(uint32_t hr=0;hr<b.rows;++hr)for(uint32_t lr=0;lr<b.cols;++lr){int g=0;Code k=dt_host_main_index(z,f,l,bid,hr,lr,true,g);
            uint64_t want=uint64_t(b.off+Code(hr)*b.cols+lr)+1;if(mv[g][size_t(k)]!=want){std::cerr<<"L2H main mismatch\n";return 542;}}}
    for(int bid=0;bid<int(l.block_blocks.size());++bid){const auto&b=l.block_blocks[bid];if(!b.valid)continue;
        for(uint32_t hr=0;hr<b.rows;++hr)for(uint32_t lr=0;lr<b.cols;++lr){int g=0;Code k=dt_host_block_index(z,f,l,bid,hr,lr,true,g);
            uint64_t want=uint64_t(b.off+Code(hr)*b.cols+lr)+1;if(bv[g][size_t(k)]!=want){std::cerr<<"L2H block mismatch\n";return 543;}}}

    dt_cpu_swap(z,mv,false,false);dt_cpu_swap(z,bv,true,false);
    for(int bid=0;bid<int(l.main_blocks.size());++bid){const auto&b=l.main_blocks[bid];if(!b.valid)continue;
        for(uint32_t hr=0;hr<b.rows;++hr)for(uint32_t lr=0;lr<b.cols;++lr){int g=0;Code k=dt_host_main_index(z,f,l,bid,hr,lr,false,g);
            uint64_t want=uint64_t(b.off+Code(hr)*b.cols+lr)+1;if(mv[g][size_t(k)]!=want){std::cerr<<"H2L main mismatch\n";return 544;}}}

    // CROSS changes topology but not occupancy on the inactive segment.  This
    // is the invariant that makes both windows completely local after a shuffle.
    uint64_t hc=0,lc=0;
    for(uint64_t op:sparse.high_closure){
        uint32_t sb=b300_sparse_closure_sblock(op),desc=b300_sparse_closure_desc(op);
        if(b300_host_high_kind(desc)!=HIGHDESC_CROSS)continue;
        uint32_t depth=(desc>>HIGHDESC_DEPTH_SHIFT)&HIGHDESC_DEPTH_MASK;
        const auto&x=l.main_blocks[sb];
        for(uint32_t lr=0;lr<x.cols;++lr){
            uint32_t c=f.low_all_codes[f.low_all_off[x.hs]+lr];uint32_t d=dt_flip_low_host(c,depth);
            if(d!=0xffffffffu&&seg_occ(c,LOW_LUT_K)!=seg_occ(d,LOW_LUT_K)){std::cerr<<"HIGH CROSS changed LOW occupancy\n";return 545;}++hc;
        }
    }
    for(uint64_t op:sparse.low_closure){
        uint32_t sb=b300_sparse_closure_sblock(op),desc=b300_sparse_closure_desc(op);
        if(b300_host_low_kind(desc)!=LOWDESC_CROSS)continue;
        uint32_t depth=(desc>>LOWDESC_DEPTH_SHIFT)&LOWDESC_DEPTH_MASK;
        const auto&x=l.main_blocks[sb];
        for(uint32_t hr=0;hr<x.rows;++hr){
            uint32_t c=f.high_all_codes[f.high_all_off[x.he]+hr];uint32_t d=dt_flip_high_host(c,depth);
            if(d!=0xffffffffu&&seg_occ(c,HIGH_LUT_K)!=seg_occ(d,HIGH_LUT_K)){std::cerr<<"LOW CROSS changed HIGH occupancy\n";return 546;}++lc;
        }
    }

    std::cout<<"b300-dual-tile-selftest OK W="<<TARGET_W<<" gpus="<<NG
             <<" main="<<l.main_size<<" block="<<l.block_size
             <<" high_cross_checks="<<hc<<" low_cross_checks="<<lc<<'\n';
    return 0;
}
