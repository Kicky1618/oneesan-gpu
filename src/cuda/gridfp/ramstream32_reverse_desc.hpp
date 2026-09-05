#pragma once

#include "../../common/gridfp_transition_reverse.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

// Correctness-first descriptor for the reflected/snake row direction.
// Keep CROSS_MAIN and CROSS_BLOCK separate until the reverse direct orbit
// structure has been classified. The final direct form can repack these later.
enum ReverseDescKind : uint8_t {
    REVDESC_INVALID = 0,
    REVDESC_MAIN = 1,
    REVDESC_BLOCK = 2,
    REVDESC_CROSS_MAIN = 3,
    REVDESC_CROSS_BLOCK = 4,
};

struct ReverseDesc {
    uint32_t rank = 0;      // destination active-half all-rank
    uint16_t block = 0;     // destination StorageBlock index
    uint8_t kind = REVDESC_INVALID;
    uint8_t depth = 0;      // inactive-half boundary flip depth for CROSS
};
static_assert(sizeof(ReverseDesc) == 8);

struct ReverseLowDescHost {
    std::vector<ReverseDesc> main_desc;
    std::vector<ReverseDesc> block_desc;
    std::array<uint32_t,64> main_base{};
    std::array<uint32_t,32> block_base{};
    uint32_t main_total=0,block_total=0;
    uint64_t observations=0,invalid=0,cross_main=0,cross_block=0;
};
struct ReverseHighDescHost {
    std::vector<ReverseDesc> main_desc;
    std::vector<ReverseDesc> block_desc;
    std::array<uint32_t,64> main_base{};
    std::array<uint32_t,32> block_base{};
    uint32_t main_total=0,block_total=0;
    uint64_t observations=0,invalid=0,cross_main=0,cross_block=0;
};

// Same inactive-half maps used by the forward CROSS machinery. Reverse-scan
// descriptors only need to recover the unique depth that maps source->dest.
static inline uint32_t reverse_desc_flip_high(uint32_t hc,uint32_t depth){
    int s=int(depth);
    for(int pos=0;pos<HIGH_LUT_K;++pos){
        MateValue v=MateValue((hc>>(2*pos))&3u);
        if(v==::L){
            if(--s==0){uint32_t z=3u<<(2*pos);return (hc&~z)|(uint32_t(R)<<(2*pos));}
        }else if(v==R)++s;
    }
    return 0xffffffffu;
}
static inline uint32_t reverse_desc_flip_low(uint32_t lc,uint32_t depth){
    int s=int(depth);
    for(int pos=LOW_LUT_K-1;pos>=0;--pos){
        MateValue v=MateValue((lc>>(2*pos))&3u);
        if(v==::L)++s;
        else if(v==R){
            if(--s==0){uint32_t z=3u<<(2*pos);return (lc&~z)|(uint32_t(::L)<<(2*pos));}
        }
    }
    return 0xffffffffu;
}
static inline uint8_t reverse_desc_high_depth(uint32_t src,uint32_t dst){
    uint8_t found=0;
    for(uint32_t d=1;d<=uint32_t(HIGH_LUT_K);++d){
        if(reverse_desc_flip_high(src,d)==dst){
            if(found){std::cerr<<"reverse LOW CROSS depth not unique\n";std::exit(230);}
            found=uint8_t(d);
        }
    }
    if(!found){std::cerr<<"reverse LOW CROSS depth missing\n";std::exit(231);}
    return found;
}
static inline uint8_t reverse_desc_low_depth(uint32_t src,uint32_t dst){
    uint8_t found=0;
    for(uint32_t d=1;d<=uint32_t(LOW_LUT_K);++d){
        if(reverse_desc_flip_low(src,d)==dst){
            if(found){std::cerr<<"reverse HIGH CROSS depth not unique\n";std::exit(232);}
            found=uint8_t(d);
        }
    }
    if(!found){std::cerr<<"reverse HIGH CROSS depth missing\n";std::exit(233);}
    return found;
}

static ReverseLowDescHost build_reverse_low_descriptors(
    const StorageFactorHost&storage,const StorageLayout&layout
){
    constexpr int L=LOW_LUT_K,H=HIGH_LUT_K;
    constexpr uint32_t LM=(1u<<(2*L))-1u;
    constexpr uint32_t HM=(1u<<(2*H))-1u;
    ReverseLowDescHost d;
    uint32_t mt=0,bt=0;
    for(size_t bid=0;bid<layout.main_blocks.size();++bid){d.main_base[bid]=mt;mt+=layout.main_blocks[bid].cols;}
    for(size_t bid=0;bid<layout.block_blocks.size();++bid){d.block_base[bid]=bt;bt+=layout.block_blocks[bid].cols;}
    d.main_total=mt;d.block_total=bt;
    d.main_desc.resize(size_t(mt)*L);d.block_desc.resize(size_t(bt)*L);

    auto representative_high=[&](int he)->uint32_t{
        uint32_t a=storage.high_all_off[he],b=storage.high_all_off[he+1];
        return a<b?storage.high_all_codes[a]:0xffffffffu;
    };

    for(int p=1;p<=L;++p){
        uint32_t pi=uint32_t(p-1);
        for(uint32_t bid=0;bid<uint32_t(layout.main_blocks.size());++bid){
            const StorageBlock&sb=layout.main_blocks[bid];
            if(!sb.valid||!sb.rows||!sb.cols)continue;
            uint32_t hc=representative_high(sb.he);if(hc==0xffffffffu)continue;
            uint32_t low0=storage.low_all_off[sb.hs];
            for(uint32_t lr=0;lr<sb.cols;++lr){
                ++d.observations;
                uint32_t lc=storage.low_all_codes[low0+lr];
                MateID m=MateID(lc)|(MateID(sb.c)<<(2*L))|(MateID(hc)<<(2*(L+1)));
                auto z=oneesan::gridfp::include_horizontal_reverse(m,TARGET_W,p);
                ReverseDesc out{};
                if(!z.valid){++d.invalid;d.main_desc[size_t(pi)*mt+d.main_base[bid]+lr]=out;continue;}
                uint32_t hc2=z.blocked?uint32_t((z.mate>>(2*L))&HM):uint32_t((z.mate>>(2*(L+1)))&HM);
                uint32_t lc2=uint32_t(z.mate)&LM;
                uint32_t packed=storage.low_packed_rank[lc2];
                if(packed==0xffffffffu){std::cerr<<"reverse LOW destination code missing\n";std::exit(234);}
                out.rank=packed>>L;
                bool cross=hc2!=hc;
                if(z.blocked){
                    int h2=seg_end_height_host(hc2,H);out.block=uint16_t(h2);
                    if(out.block>=layout.block_blocks.size()||out.rank>=layout.block_blocks[out.block].cols)std::exit(235);
                    out.kind=cross?REVDESC_CROSS_BLOCK:REVDESC_BLOCK;
                    if(cross){out.depth=reverse_desc_high_depth(hc,hc2);++d.cross_block;}
                }else{
                    int he2=seg_end_height_host(hc2,H),cv2=int(mget(z.mate,L));
                    out.block=uint16_t(3*he2+cv2);
                    if(out.block>=layout.main_blocks.size()||out.rank>=layout.main_blocks[out.block].cols)std::exit(236);
                    out.kind=cross?REVDESC_CROSS_MAIN:REVDESC_MAIN;
                    if(cross){out.depth=reverse_desc_high_depth(hc,hc2);++d.cross_main;}
                }
                d.main_desc[size_t(pi)*mt+d.main_base[bid]+lr]=out;
            }
        }

        for(uint32_t bid=0;bid<uint32_t(layout.block_blocks.size());++bid){
            const StorageBlock&sb=layout.block_blocks[bid];
            if(!sb.valid||!sb.rows||!sb.cols)continue;
            uint32_t hc=representative_high(sb.he);if(hc==0xffffffffu)continue;
            uint32_t low0=storage.low_all_off[sb.hs];
            for(uint32_t lr=0;lr<sb.cols;++lr){
                uint32_t lc=storage.low_all_codes[low0+lr];
                MateID m=MateID(lc)|(MateID(hc)<<(2*L));
                MateID z=oneesan::gridfp::blocked_exclude_reverse(m,TARGET_W,p);
                uint32_t hc2=uint32_t((z>>(2*(L+1)))&HM);
                if(hc2!=hc){std::cerr<<"reverse LOW blocked changed HIGH\n";std::exit(237);}
                uint32_t lc2=uint32_t(z)&LM,packed=storage.low_packed_rank[lc2];
                if(packed==0xffffffffu)std::exit(238);
                ReverseDesc out{};out.kind=REVDESC_MAIN;out.rank=packed>>L;
                int he2=seg_end_height_host(hc2,H),cv2=int(mget(z,L));out.block=uint16_t(3*he2+cv2);
                if(out.block>=layout.main_blocks.size()||out.rank>=layout.main_blocks[out.block].cols)std::exit(239);
                d.block_desc[size_t(pi)*bt+d.block_base[bid]+lr]=out;
            }
        }
    }
    std::cerr<<"reverse_low_desc active="<<d.main_total
             <<" observations="<<d.observations<<" invalid="<<d.invalid
             <<" cross_main="<<d.cross_main<<" cross_block="<<d.cross_block
             <<" mib="<<double((d.main_desc.size()+d.block_desc.size())*sizeof(ReverseDesc))/(1<<20)<<'\n';
    return d;
}

static ReverseHighDescHost build_reverse_high_descriptors(
    const StorageFactorHost&storage,const StorageLayout&layout
){
    constexpr int L=LOW_LUT_K,H=HIGH_LUT_K;
    constexpr uint32_t LM=(1u<<(2*L))-1u;
    constexpr uint32_t HM=(1u<<(2*H))-1u;
    ReverseHighDescHost d;
    uint32_t mt=0,bt=0;
    for(size_t bid=0;bid<layout.main_blocks.size();++bid){d.main_base[bid]=mt;mt+=layout.main_blocks[bid].rows;}
    for(size_t bid=0;bid<layout.block_blocks.size();++bid){d.block_base[bid]=bt;bt+=layout.block_blocks[bid].rows;}
    d.main_total=mt;d.block_total=bt;
    d.main_desc.resize(size_t(mt)*H);d.block_desc.resize(size_t(bt)*H);

    auto representative_low=[&](int hs)->uint32_t{
        uint32_t a=storage.low_all_off[hs],b=storage.low_all_off[hs+1];
        return a<b?storage.low_all_codes[a]:0xffffffffu;
    };

    for(int p=L+1;p<TARGET_W;++p){
        uint32_t pi=uint32_t(p-(L+1));
        for(uint32_t bid=0;bid<uint32_t(layout.main_blocks.size());++bid){
            const StorageBlock&sb=layout.main_blocks[bid];
            if(!sb.valid||!sb.rows||!sb.cols)continue;
            uint32_t lc=representative_low(sb.hs);if(lc==0xffffffffu)continue;
            uint32_t high0=storage.high_all_off[sb.he];
            for(uint32_t hr=0;hr<sb.rows;++hr){
                ++d.observations;
                uint32_t hc=storage.high_all_codes[high0+hr];
                MateID m=MateID(lc)|(MateID(sb.c)<<(2*L))|(MateID(hc)<<(2*(L+1)));
                auto z=oneesan::gridfp::include_horizontal_reverse(m,TARGET_W,p);
                ReverseDesc out{};
                if(!z.valid){++d.invalid;d.main_desc[size_t(pi)*mt+d.main_base[bid]+hr]=out;continue;}
                uint32_t lc2=uint32_t(z.mate)&LM;
                uint32_t hc2=z.blocked?uint32_t((z.mate>>(2*L))&HM):uint32_t((z.mate>>(2*(L+1)))&HM);
                uint32_t packed=storage.high_packed_rank[hc2];
                if(packed==0xffffffffu){std::cerr<<"reverse HIGH destination code missing\n";std::exit(240);}
                out.rank=packed>>H;
                bool cross=lc2!=lc;
                if(z.blocked){
                    int h2=seg_end_height_host(hc2,H);out.block=uint16_t(h2);
                    if(out.block>=layout.block_blocks.size()||out.rank>=layout.block_blocks[out.block].rows)std::exit(241);
                    out.kind=cross?REVDESC_CROSS_BLOCK:REVDESC_BLOCK;
                    if(cross){out.depth=reverse_desc_low_depth(lc,lc2);++d.cross_block;}
                }else{
                    int he2=seg_end_height_host(hc2,H),cv2=int(mget(z.mate,L));out.block=uint16_t(3*he2+cv2);
                    if(out.block>=layout.main_blocks.size()||out.rank>=layout.main_blocks[out.block].rows)std::exit(242);
                    out.kind=cross?REVDESC_CROSS_MAIN:REVDESC_MAIN;
                    if(cross){out.depth=reverse_desc_low_depth(lc,lc2);++d.cross_main;}
                }
                d.main_desc[size_t(pi)*mt+d.main_base[bid]+hr]=out;
            }
        }

        for(uint32_t bid=0;bid<uint32_t(layout.block_blocks.size());++bid){
            const StorageBlock&sb=layout.block_blocks[bid];
            if(!sb.valid||!sb.rows||!sb.cols)continue;
            uint32_t lc=representative_low(sb.hs);if(lc==0xffffffffu)continue;
            uint32_t high0=storage.high_all_off[sb.he];
            for(uint32_t hr=0;hr<sb.rows;++hr){
                uint32_t hc=storage.high_all_codes[high0+hr];
                MateID m=MateID(lc)|(MateID(hc)<<(2*L));
                MateID z=oneesan::gridfp::blocked_exclude_reverse(m,TARGET_W,p);
                uint32_t lc2=uint32_t(z)&LM;
                if(lc2!=lc){std::cerr<<"reverse HIGH blocked changed LOW\n";std::exit(243);}
                uint32_t hc2=uint32_t((z>>(2*(L+1)))&HM),packed=storage.high_packed_rank[hc2];
                if(packed==0xffffffffu)std::exit(244);
                ReverseDesc out{};out.kind=REVDESC_MAIN;out.rank=packed>>H;
                int he2=seg_end_height_host(hc2,H),cv2=int(mget(z,L));out.block=uint16_t(3*he2+cv2);
                if(out.block>=layout.main_blocks.size()||out.rank>=layout.main_blocks[out.block].rows)std::exit(245);
                d.block_desc[size_t(pi)*bt+d.block_base[bid]+hr]=out;
            }
        }
    }
    std::cerr<<"reverse_high_desc active="<<d.main_total
             <<" observations="<<d.observations<<" invalid="<<d.invalid
             <<" cross_main="<<d.cross_main<<" cross_block="<<d.cross_block
             <<" mib="<<double((d.main_desc.size()+d.block_desc.size())*sizeof(ReverseDesc))/(1<<20)<<'\n';
    return d;
}
