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
static bool rds_eq(const ReverseDesc&a,const ReverseDesc&b){
    if(a.kind!=b.kind)return false;
    if(a.kind==REVDESC_INVALID)return true;
    return a.rank==b.rank&&a.block==b.block&&a.depth==b.depth;
}
static void rds_fail(const char*side,const char*src,MateID m,int p,const ReverseDesc&got,const ReverseDesc&want){
    std::cerr<<"FAIL reverse-desc "<<side<<' '<<src<<" m="<<m<<" p="<<p
             <<" got=("<<unsigned(got.kind)<<','<<got.block<<','<<got.rank<<','<<unsigned(got.depth)<<')'
             <<" want=("<<unsigned(want.kind)<<','<<want.block<<','<<want.rank<<','<<unsigned(want.depth)<<')'<<'\n';
    std::exit(10);
}

static ReverseDesc rds_expected_low_main(MateID m,int p,const StorageFactorHost&s,const StorageLayout&layout){
    constexpr int L=LOW_LUT_K,H=HIGH_LUT_K;
    constexpr uint32_t LM=(1u<<(2*L))-1u,HM=(1u<<(2*H))-1u;
    auto z=oneesan::gridfp::include_horizontal_reverse(m,TARGET_W,p);ReverseDesc out{};
    if(!z.valid)return out;
    uint32_t hc=uint32_t((m>>(2*(L+1)))&HM);
    uint32_t hc2=z.blocked?uint32_t((z.mate>>(2*L))&HM):uint32_t((z.mate>>(2*(L+1)))&HM);
    uint32_t lc2=uint32_t(z.mate)&LM,packed=s.low_packed_rank[lc2];if(packed==0xffffffffu)std::exit(250);
    out.rank=packed>>L;bool cross=hc2!=hc;
    if(z.blocked){out.block=uint16_t(seg_end_height_host(hc2,H));out.kind=cross?REVDESC_CROSS_BLOCK:REVDESC_BLOCK;if(cross)out.depth=reverse_desc_high_depth(hc,hc2);if(out.rank>=layout.block_blocks[out.block].cols)std::exit(251);}
    else{int he2=seg_end_height_host(hc2,H),cv2=int(mget(z.mate,L));out.block=uint16_t(3*he2+cv2);out.kind=cross?REVDESC_CROSS_MAIN:REVDESC_MAIN;if(cross)out.depth=reverse_desc_high_depth(hc,hc2);if(out.rank>=layout.main_blocks[out.block].cols)std::exit(252);}
    return out;
}
static ReverseDesc rds_expected_high_main(MateID m,int p,const StorageFactorHost&s,const StorageLayout&layout){
    constexpr int L=LOW_LUT_K,H=HIGH_LUT_K;
    constexpr uint32_t LM=(1u<<(2*L))-1u,HM=(1u<<(2*H))-1u;
    auto z=oneesan::gridfp::include_horizontal_reverse(m,TARGET_W,p);ReverseDesc out{};
    if(!z.valid)return out;
    uint32_t lc=uint32_t(m)&LM,lc2=uint32_t(z.mate)&LM;
    uint32_t hc2=z.blocked?uint32_t((z.mate>>(2*L))&HM):uint32_t((z.mate>>(2*(L+1)))&HM);
    uint32_t packed=s.high_packed_rank[hc2];if(packed==0xffffffffu)std::exit(253);
    out.rank=packed>>H;bool cross=lc2!=lc;
    if(z.blocked){out.block=uint16_t(seg_end_height_host(hc2,H));out.kind=cross?REVDESC_CROSS_BLOCK:REVDESC_BLOCK;if(cross)out.depth=reverse_desc_low_depth(lc,lc2);if(out.rank>=layout.block_blocks[out.block].rows)std::exit(254);}
    else{int he2=seg_end_height_host(hc2,H),cv2=int(mget(z.mate,L));out.block=uint16_t(3*he2+cv2);out.kind=cross?REVDESC_CROSS_MAIN:REVDESC_MAIN;if(cross)out.depth=reverse_desc_low_depth(lc,lc2);if(out.rank>=layout.main_blocks[out.block].rows)std::exit(255);}
    return out;
}
static ReverseDesc rds_expected_low_block(MateID m,int p,const StorageFactorHost&s,const StorageLayout&layout){
    constexpr int L=LOW_LUT_K,H=HIGH_LUT_K;
    constexpr uint32_t LM=(1u<<(2*L))-1u,HM=(1u<<(2*H))-1u;
    MateID z=oneesan::gridfp::blocked_exclude_reverse(m,TARGET_W,p);
    uint32_t hc=uint32_t((m>>(2*L))&HM),hc2=uint32_t((z>>(2*(L+1)))&HM);if(hc!=hc2)std::exit(256);
    uint32_t lc2=uint32_t(z)&LM,packed=s.low_packed_rank[lc2];ReverseDesc out{};out.kind=REVDESC_MAIN;out.rank=packed>>L;
    int he2=seg_end_height_host(hc2,H),cv2=int(mget(z,L));out.block=uint16_t(3*he2+cv2);if(out.rank>=layout.main_blocks[out.block].cols)std::exit(257);return out;
}
static ReverseDesc rds_expected_high_block(MateID m,int p,const StorageFactorHost&s,const StorageLayout&layout){
    constexpr int L=LOW_LUT_K,H=HIGH_LUT_K;
    constexpr uint32_t LM=(1u<<(2*L))-1u,HM=(1u<<(2*H))-1u;
    MateID z=oneesan::gridfp::blocked_exclude_reverse(m,TARGET_W,p);
    uint32_t lc=uint32_t(m)&LM,lc2=uint32_t(z)&LM;if(lc!=lc2)std::exit(258);
    uint32_t hc2=uint32_t((z>>(2*(L+1)))&HM),packed=s.high_packed_rank[hc2];ReverseDesc out{};out.kind=REVDESC_MAIN;out.rank=packed>>H;
    int he2=seg_end_height_host(hc2,H),cv2=int(mget(z,L));out.block=uint16_t(3*he2+cv2);if(out.rank>=layout.main_blocks[out.block].rows)std::exit(259);return out;
}

int main(){
    constexpr int W=TARGET_W,L=LOW_LUT_K,H=HIGH_LUT_K;
    constexpr uint32_t LM=(1u<<(2*L))-1u,HM=(1u<<(2*H))-1u;
    static_assert(W<=12,"reverse descriptor selftest intentionally uses small width");
    static_assert(L+H+1==W);
    build_full_dp();G_FACTOR=build_factor_tables();StorageFactorHost storage=build_storage_factor_tables(G_FACTOR);StorageLayout layout=build_storage_layout(storage);
    ReverseLowDescHost low=build_reverse_low_descriptors(storage,layout);ReverseHighDescHost high=build_reverse_high_descriptors(storage,layout);
    auto ms=rds_states(W),bs=rds_states(W-1);if(ms.size()!=layout.main_size||bs.size()!=layout.block_size)return 2;
    uint64_t mc=0,bc=0;
    for(MateID m:ms){
        uint32_t bid=rds_main_bid(m),lc=uint32_t(m)&LM,hc=uint32_t((m>>(2*(L+1)))&HM);uint32_t lr=storage.low_packed_rank[lc]>>L,hr=storage.high_packed_rank[hc]>>H;
        for(int p=1;p<=L;++p){uint32_t pi=uint32_t(p-1);ReverseDesc got=low.main_desc[size_t(pi)*low.main_total+low.main_base[bid]+lr],want=rds_expected_low_main(m,p,storage,layout);if(!rds_eq(got,want))rds_fail("LOW","main",m,p,got,want);++mc;}
        for(int p=L+1;p<W;++p){uint32_t pi=uint32_t(p-(L+1));ReverseDesc got=high.main_desc[size_t(pi)*high.main_total+high.main_base[bid]+hr],want=rds_expected_high_main(m,p,storage,layout);if(!rds_eq(got,want))rds_fail("HIGH","main",m,p,got,want);++mc;}
    }
    for(MateID m:bs){
        uint32_t bid=rds_block_bid(m),lc=uint32_t(m)&LM,hc=uint32_t((m>>(2*L))&HM);uint32_t lr=storage.low_packed_rank[lc]>>L,hr=storage.high_packed_rank[hc]>>H;
        for(int p=1;p<=L;++p){uint32_t pi=uint32_t(p-1);ReverseDesc got=low.block_desc[size_t(pi)*low.block_total+low.block_base[bid]+lr],want=rds_expected_low_block(m,p,storage,layout);if(!rds_eq(got,want))rds_fail("LOW","block",m,p,got,want);++bc;}
        for(int p=L+1;p<W;++p){uint32_t pi=uint32_t(p-(L+1));ReverseDesc got=high.block_desc[size_t(pi)*high.block_total+high.block_base[bid]+hr],want=rds_expected_high_block(m,p,storage,layout);if(!rds_eq(got,want))rds_fail("HIGH","block",m,p,got,want);++bc;}
    }
    std::cout<<"reverse-desc-selftest OK W="<<W<<" main_checks="<<mc<<" block_checks="<<bc
             <<" low_cross_main="<<low.cross_main<<" low_cross_block="<<low.cross_block
             <<" high_cross_main="<<high.cross_main<<" high_cross_block="<<high.cross_block<<'\n';
    return 0;
}
