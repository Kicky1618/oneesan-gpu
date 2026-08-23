#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <vector>

#define RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../oneesan_cuda_gridfp_ramstream32_factorized_bidesc_compact.cu"
#undef RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../ramstream32_high_orbit.cuh"
#include "../ramstream32_cpu_low_inplace.hpp"
#include "../ramstream32_b300_dual_tile_pruned_peer.cuh"

namespace {

using Arena = std::vector<std::vector<uint64_t>>;

static Arena make_arena(const std::array<Code,MAXGPU>& n,int ngpu){
    Arena a(ngpu);for(int g=0;g<ngpu;++g)a[g].assign(size_t(n[g]),0);return a;
}

static void full_swap_cpu(const B300DualTileHost&z,Arena&v,bool blocked,bool l2h){
    const auto&sz=blocked?z.pair_block_size:z.pair_main_size;
    const auto&bs=blocked?z.block_slot_base:z.main_slot_base;
    for(int a=0;a<z.ngpu;++a)for(int b=a+1;b<z.ngpu;++b){
        Code na=l2h?sz[b][a]:sz[a][b],nb=l2h?sz[a][b]:sz[b][a];
        std::vector<uint64_t> ta(size_t(na)),tb(size_t(nb));
        for(Code i=0;i<na;++i)ta[size_t(i)]=v[a][size_t(bs[a][b]+i)];
        for(Code i=0;i<nb;++i)tb[size_t(i)]=v[b][size_t(bs[b][a]+i)];
        for(Code i=0;i<na;++i)v[b][size_t(bs[b][a]+i)]=ta[size_t(i)];
        for(Code i=0;i<nb;++i)v[a][size_t(bs[a][b]+i)]=tb[size_t(i)];
    }
}

static void pruned_swap_cpu(
    const B300DualTileHost&z,const StorageLayout&l,Arena&v,bool blocked,bool l2h,
    const B300DualReachStage&stage
){
    const auto&sz=blocked?z.pair_block_size:z.pair_main_size;
    const auto&bs=blocked?z.block_slot_base:z.main_slot_base;
    std::vector<Code> bounds;
    for(int a=0;a<z.ngpu;++a)for(int b=a+1;b<z.ngpu;++b){
        int ahi,alo,bhi,blo;
        if(l2h){ahi=b;alo=a;bhi=a;blo=b;}
        else {ahi=a;alo=b;bhi=b;blo=a;}
        b300_dt_pruned_boundaries(z,l,blocked,ahi,alo,bhi,blo,bounds);
        Code limit=std::max(sz[ahi][alo],sz[bhi][blo]);
        for(size_t k=0;k+1<bounds.size();++k){
            Code x=bounds[k],y=std::min(bounds[k+1],limit);if(x>=y)continue;
            bool aa=x<sz[ahi][alo]&&b300_dt_pruned_stream_active_at(z,l,stage,blocked,ahi,alo,x);
            bool bb=x<sz[bhi][blo]&&b300_dt_pruned_stream_active_at(z,l,stage,blocked,bhi,blo,x);
            for(Code i=x;i<y;++i){
                uint64_t&A=v[a][size_t(bs[a][b]+i)],&B=v[b][size_t(bs[b][a]+i)];
                if(aa&&bb){uint64_t t=A;A=B;B=t;}
                else if(aa){B=A;A=0;}
                else if(bb){A=B;B=0;}
            }
        }
    }
}

static uint64_t value_for(bool blocked,int bid,int hi,int lo,Code q){
    uint64_t x=uint64_t(q)+1;
    x^=uint64_t(bid+1)*0x9e3779b97f4a7c15ULL;
    x^=uint64_t(hi+3)<<47;x^=uint64_t(lo+5)<<39;
    if(blocked)x^=0xd1b54a32d192ed03ULL;
    return x?x:1;
}

static void materialize_stage(
    const B300DualTileHost&z,const StorageLayout&l,const B300DualReachStage&stage,
    Arena&v,bool blocked,bool high_orientation
){
    const int nb=blocked?int(l.block_blocks.size()):int(l.main_blocks.size());
    const auto&bs=blocked?z.block_slot_base:z.main_slot_base;
    for(int hi=0;hi<z.ngpu;++hi)for(int lo=0;lo<z.ngpu;++lo){
        int owner=high_orientation?hi:lo,peer=high_orientation?lo:hi;
        for(int bid=0;bid<nb;++bid){
            const auto&b=blocked?l.block_blocks[bid]:l.main_blocks[bid];
            Code n=b300_dt_pruned_seg_len(z,b,hi,lo);if(!n)continue;
            bool active=blocked?b300_dt_reach_block_active(stage,uint32_t(bid),hi,lo,z.ngpu)
                               :b300_dt_reach_main_active(stage,uint32_t(bid),hi,lo,z.ngpu);
            if(!active)continue;
            Code off=b300_dt_pruned_seg_off(z,blocked,nb,hi,lo,bid);
            for(Code q=0;q<n;++q)v[owner][size_t(bs[owner][peer]+off+q)]=value_for(blocked,bid,hi,lo,q);
        }
    }
}

static bool compare_logical_target(
    const B300DualTileHost&z,const StorageLayout&l,const Arena&a,const Arena&b,
    bool blocked,bool high_orientation
){
    const int nb=blocked?int(l.block_blocks.size()):int(l.main_blocks.size());
    const auto&bs=blocked?z.block_slot_base:z.main_slot_base;
    for(int hi=0;hi<z.ngpu;++hi)for(int lo=0;lo<z.ngpu;++lo){
        int owner=high_orientation?hi:lo,peer=high_orientation?lo:hi;
        for(int bid=0;bid<nb;++bid){
            const auto&blk=blocked?l.block_blocks[bid]:l.main_blocks[bid];
            Code n=b300_dt_pruned_seg_len(z,blk,hi,lo);if(!n)continue;
            Code off=b300_dt_pruned_seg_off(z,blocked,nb,hi,lo,bid);
            for(Code q=0;q<n;++q){Code ix=bs[owner][peer]+off+q;
                if(a[owner][size_t(ix)]!=b[owner][size_t(ix)]){
                    std::cerr<<"logical target mismatch blocked="<<blocked
                             <<" high="<<high_orientation<<" bid="<<bid
                             <<" hi="<<hi<<" lo="<<lo<<" q="<<q<<'\n';return false;}
            }
        }
    }
    return true;
}

} // namespace

int main(){
    constexpr int NG=4;
    build_full_dp();G_FACTOR=build_factor_tables();
    StorageFactorHost f=build_storage_factor_tables(G_FACTOR);StorageLayout l=build_storage_layout(f);
    LowDescHost ld=build_low_descriptors(f,l);HighDescHost hd=build_high_descriptors(f,l);
    LowOrbitHost lo=build_cpu_low_orbit(f,l,ld);HighOrbitHost ho=build_high_orbit(f,l);
    B300SparseActionsHost sparse=build_b300_sparse_actions(l,ld,lo,hd,ho);
    B300DualTileHost z=build_b300_dual_tile_layout(f,l,NG);
    B300DualReachSchedule reach=build_b300_dual_reach_schedule(sparse,f,l,z);
    if(reach.l2h.size()!=TARGET_W||reach.h2l.size()!=TARGET_W-1)return 660;

    for(int row=0;row<TARGET_W;++row){
        Arena m=make_arena(z.main_count,NG),b=make_arena(z.block_count,NG);
        materialize_stage(z,l,reach.l2h[row],m,false,false);
        materialize_stage(z,l,reach.l2h[row],b,true,false);
        Arena mf=m,bf=b,mp=m,bp=b;
        full_swap_cpu(z,mf,false,true);full_swap_cpu(z,bf,true,true);
        pruned_swap_cpu(z,l,mp,false,true,reach.l2h[row]);
        pruned_swap_cpu(z,l,bp,true,true,reach.l2h[row]);
        if(!compare_logical_target(z,l,mf,mp,false,true))return 661;
        if(!compare_logical_target(z,l,bf,bp,true,true))return 662;

        if(row+1<TARGET_W){
            Arena x=make_arena(z.main_count,NG);
            materialize_stage(z,l,reach.h2l[row],x,false,true);
            Arena xf=x,xp=x;
            full_swap_cpu(z,xf,false,false);
            pruned_swap_cpu(z,l,xp,false,false,reach.h2l[row]);
            if(!compare_logical_target(z,l,xf,xp,false,false))return 663;
        }
    }
    std::cout<<"b300-dual-tile-pruned-selftest OK W="<<TARGET_W
             <<" gpus="<<NG<<" rows="<<TARGET_W<<'\n';
    return 0;
}
