#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#define RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../oneesan_cuda_gridfp_ramstream32_factorized_bidesc_compact.cu"
#undef RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../ramstream32_reverse_desc.hpp"

static void rds_enum_rec(int pos,int h,MateID m,std::vector<MateID>&out){
    if(pos<0){if(h==0)out.push_back(m);return;}
    rds_enum_rec(pos-1,h,mset(m,pos,N),out);
    if(h>0)rds_enum_rec(pos-1,h-1,mset(m,pos,R),out);
    rds_enum_rec(pos-1,h+1,mset(m,pos,::L),out);
}
static std::vector<MateID> rds_states(int w){std::vector<MateID>v;rds_enum_rec(w-1,1,0,v);return v;}

static uint32_t rds_main_bid(MateID m){
    constexpr int L=LOW_LUT_K,H=HIGH_LUT_K;
    constexpr uint32_t HM=(1u<<(2*H))-1u;
    uint32_t hc=uint32_t((m>>(2*(L+1)))&HM);
    return uint32_t(3*seg_end_height_host(hc,H)+int(mget(m,L)));
}
static uint32_t rds_block_bid(MateID m){
    constexpr int L=LOW_LUT_K,H=HIGH_LUT_K;
    constexpr uint32_t HM=(1u<<(2*H))-1u;
    uint32_t hc=uint32_t((m>>(2*L))&HM);
    return uint32_t(seg_end_height_host(hc,H));
}

static oneesan::gridfp::IncludeResult rds_predict_low_main(
    MateID m,int p,const ReverseLowDescHost&d,
    const StorageFactorHost&s,const StorageLayout&layout
){
    constexpr int L=LOW_LUT_K,H=HIGH_LUT_K;
    constexpr uint32_t LM=(1u<<(2*L))-1u,HM=(1u<<(2*H))-1u;
    uint32_t bid=rds_main_bid(m),lc=uint32_t(m)&LM;
    uint32_t lp=s.low_packed_rank[lc];if(lp==0xffffffffu)std::exit(250);
    uint32_t lr=lp>>L,pi=uint32_t(p-1);
    ReverseDesc x=d.main_desc[size_t(pi)*d.main_total+d.main_base[bid]+lr];
    oneesan::gridfp::IncludeResult z{};
    if(x.kind==REVDESC_INVALID)return z;
    bool blocked=x.kind==REVDESC_BLOCK||x.kind==REVDESC_CROSS_BLOCK;
    bool cross=x.kind==REVDESC_CROSS_MAIN||x.kind==REVDESC_CROSS_BLOCK;
    uint32_t hc=uint32_t((m>>(2*(L+1)))&HM);
    uint32_t hc2=cross?reverse_desc_flip_high(hc,x.depth):hc;
    if(hc2==0xffffffffu)std::exit(251);
    if(blocked){
        const auto&b=layout.block_blocks[x.block];
        uint32_t lc2=s.low_all_codes[s.low_all_off[b.hs]+x.rank];
        z.mate=MateID(lc2)|(MateID(hc2)<<(2*L));z.valid=true;z.blocked=true;
    }else{
        const auto&b=layout.main_blocks[x.block];
        uint32_t lc2=s.low_all_codes[s.low_all_off[b.hs]+x.rank];
        z.mate=MateID(lc2)|(MateID(b.c)<<(2*L))|(MateID(hc2)<<(2*(L+1)));z.valid=true;
    }
    return z;
}

static oneesan::gridfp::IncludeResult rds_predict_high_main(
    MateID m,int p,const ReverseHighDescHost&d,
    const StorageFactorHost&s,const StorageLayout&layout
){
    constexpr int L=LOW_LUT_K,H=HIGH_LUT_K;
    constexpr uint32_t LM=(1u<<(2*L))-1u,HM=(1u<<(2*H))-1u;
    uint32_t bid=rds_main_bid(m),hc=uint32_t((m>>(2*(L+1)))&HM);
    uint32_t hp=s.high_packed_rank[hc];if(hp==0xffffffffu)std::exit(252);
    uint32_t hr=hp>>H,pi=uint32_t(p-(L+1));
    ReverseDesc x=d.main_desc[size_t(pi)*d.main_total+d.main_base[bid]+hr];
    oneesan::gridfp::IncludeResult z{};
    if(x.kind==REVDESC_INVALID)return z;
    bool blocked=x.kind==REVDESC_BLOCK||x.kind==REVDESC_CROSS_BLOCK;
    bool cross=x.kind==REVDESC_CROSS_MAIN||x.kind==REVDESC_CROSS_BLOCK;
    uint32_t lc=uint32_t(m)&LM;
    uint32_t lc2=cross?reverse_desc_flip_low(lc,x.depth):lc;
    if(lc2==0xffffffffu)std::exit(253);
    if(blocked){
        const auto&b=layout.block_blocks[x.block];
        uint32_t hc2=s.high_all_codes[s.high_all_off[b.he]+x.rank];
        z.mate=MateID(lc2)|(MateID(hc2)<<(2*L));z.valid=true;z.blocked=true;
    }else{
        const auto&b=layout.main_blocks[x.block];
        uint32_t hc2=s.high_all_codes[s.high_all_off[b.he]+x.rank];
        z.mate=MateID(lc2)|(MateID(b.c)<<(2*L))|(MateID(hc2)<<(2*(L+1)));z.valid=true;
    }
    return z;
}

static MateID rds_predict_low_block(
    MateID m,int p,const ReverseLowDescHost&d,
    const StorageFactorHost&s,const StorageLayout&layout
){
    constexpr int L=LOW_LUT_K,H=HIGH_LUT_K;
    constexpr uint32_t LM=(1u<<(2*L))-1u,HM=(1u<<(2*H))-1u;
    uint32_t bid=rds_block_bid(m),lc=uint32_t(m)&LM,lp=s.low_packed_rank[lc];
    uint32_t lr=lp>>L,pi=uint32_t(p-1);ReverseDesc x=d.block_desc[size_t(pi)*d.block_total+d.block_base[bid]+lr];
    if(x.kind!=REVDESC_MAIN)std::exit(254);
    uint32_t hc=uint32_t((m>>(2*L))&HM);const auto&b=layout.main_blocks[x.block];
    uint32_t lc2=s.low_all_codes[s.low_all_off[b.hs]+x.rank];
    return MateID(lc2)|(MateID(b.c)<<(2*L))|(MateID(hc)<<(2*(L+1)));
}

static MateID rds_predict_high_block(
    MateID m,int p,const ReverseHighDescHost&d,
    const StorageFactorHost&s,const StorageLayout&layout
){
    constexpr int L=LOW_LUT_K,H=HIGH_LUT_K;
    constexpr uint32_t LM=(1u<<(2*L))-1u,HM=(1u<<(2*H))-1u;
    uint32_t bid=rds_block_bid(m),hc=uint32_t((m>>(2*L))&HM,hp=s.high_packed_rank[hc];
    uint32_t hr=hp>>H,pi=uint32_t(p-(L+1));ReverseDesc x=d.block_desc[size_t(pi)*d.block_total+d.block_base[bid]+hr];
    if(x.kind!=REVDESC_MAIN)std::exit(255);
    uint32_t lc=uint32_t(m)&LM;const auto&b=layout.main_blocks[x.block];
    uint32_t hc2=s.high_all_codes[s.high_all_off[b.he]+x.rank];
    return MateID(lc)|(MateID(b.c)<<(2*L))|(MateID(hc2)<<(2*(L+1)));
}

static bool same(const oneesan::gridfp::IncludeResult&a,const oneesan::gridfp::IncludeResult&b){
    return a.valid==b.valid&&a.blocked==b.blocked&&(!a.valid||a.mate==b.mate);
}

int main(){
    constexpr int W=TARGET_W,L=LOW_LUT_K;
    static_assert(W<=12,"reverse descriptor selftest intentionally uses small width");
    static_assert(LOW_LUT_K+HIGH_LUT_K+1==TARGET_W);
    build_full_dp();G_FACTOR=build_factor_tables();
    StorageFactorHost storage=build_storage_factor_tables(G_FACTOR);StorageLayout layout=build_storage_layout(storage);
    ReverseLowDescHost low=build_reverse_low_descriptors(storage,layout);
    ReverseHighDescHost high=build_reverse_high_descriptors(storage,layout);
    auto ms=rds_states(W),bs=rds_states(W-1);
    if(ms.size()!=layout.main_size||bs.size()!=layout.block_size)return 2;
    uint64_t checked_main=0,checked_block=0;
    for(MateID m:ms){
        for(int p=1;p<=L;++p){auto want=oneesan::gridfp::include_horizontal_reverse(m,W,p);auto got=rds_predict_low_main(m,p,low,storage,layout);if(!same(want,got)){std::cerr<<"FAIL reverse LOW main m="<<m<<" p="<<p<<'\n';return 10;}++checked_main;}
        for(int p=L+1;p<W;++p){auto want=oneesan::gridfp::include_horizontal_reverse(m,W,p);auto got=rds_predict_high_main(m,p,high,storage,layout);if(!same(want,got)){std::cerr<<"FAIL reverse HIGH main m="<<m<<" p="<<p<<'\n';return 11;}++checked_main;}
    }
    for(MateID m:bs){
        for(int p=1;p<=L;++p){MateID want=oneesan::gridfp::blocked_exclude_reverse(m,W,p),got=rds_predict_low_block(m,p,low,storage,layout);if(want!=got){std::cerr<<"FAIL reverse LOW block m="<<m<<" p="<<p<<'\n';return 12;}++checked_block;}
        for(int p=L+1;p<W;++p){MateID want=oneesan::gridfp::blocked_exclude_reverse(m,W,p),got=rds_predict_high_block(m,p,high,storage,layout);if(want!=got){std::cerr<<"FAIL reverse HIGH block m="<<m<<" p="<<p<<'\n';return 13;}++checked_block;}
    }
    std::cout<<"reverse-desc-selftest OK W="<<W
             <<" main_checks="<<checked_main<<" block_checks="<<checked_block
             <<" low_cross_main="<<low.cross_main<<" low_cross_block="<<low.cross_block
             <<" high_cross_main="<<high.cross_main<<" high_cross_block="<<high.cross_block
             <<"\n";
    return 0;
}
