#pragma once

#include "ramstream32_reverse_desc.hpp"

#include <cstdint>
#include <cstdlib>
#include <iostream>

// Include after ramstream32_lowdesc.cuh and ramstream32_highdesc.cuh.
// Reflection exchanges the structural roles of the two windows:
//  * reverse LOW is conjugate to a forward HIGH window, so every CROSS closes
//    into blocked storage;
//  * reverse HIGH is conjugate to a forward LOW window, so CROSS can return to
//    main storage only at the reflected p==1 boundary, i.e. original p==W-1.
// Hence target storage for CROSS is recoverable from p and no extra descriptor
// bit is needed. We can reuse the existing 32-bit LOW/HIGH encodings.

static void validate_reverse_low_cross_targets(const ReverseLowDescHost&low){
    if(low.cross_main){
        std::cerr<<"reverse LOW unexpectedly has CROSS_MAIN count="<<low.cross_main<<'\n';
        std::exit(260);
    }
    for(const ReverseDesc&x:low.main_desc)if(x.kind==REVDESC_CROSS_MAIN)std::exit(261);
}

static void validate_reverse_high_cross_targets(const ReverseHighDescHost&high){
    for(int p=LOW_LUT_K+1;p<TARGET_W;++p){
        uint32_t pi=uint32_t(p-(LOW_LUT_K+1));
        size_t a=size_t(pi)*high.main_total,b=a+high.main_total;
        for(size_t i=a;i<b;++i){
            const ReverseDesc&x=high.main_desc[i];
            if(x.kind==REVDESC_CROSS_MAIN&&p!=TARGET_W-1){
                std::cerr<<"reverse HIGH CROSS_MAIN away from boundary p="<<p<<'\n';std::exit(262);
            }
            if(x.kind==REVDESC_CROSS_BLOCK&&p==TARGET_W-1){
                std::cerr<<"reverse HIGH CROSS_BLOCK at main-return boundary\n";std::exit(263);
            }
        }
    }
}

static LowDescHost pack_reverse_low_descriptors(const ReverseLowDescHost&r){
    validate_reverse_low_cross_targets(r);
    LowDescHost out;
    out.main_base=r.main_base;out.block_base=r.block_base;
    out.main_total=r.main_total;out.block_total=r.block_total;
    out.main_observations=r.observations;out.main_cross=r.cross_block;out.main_invalid=r.invalid;
    out.main_desc.resize(r.main_desc.size());out.block_desc.resize(r.block_desc.size());
    auto pack=[](const ReverseDesc&x)->uint32_t{
        switch(x.kind){
        case REVDESC_INVALID:return lowdesc_pack(LOWDESC_INVALID,0,0);
        case REVDESC_MAIN:return lowdesc_pack(LOWDESC_MAIN,x.block,x.rank);
        case REVDESC_BLOCK:return lowdesc_pack(LOWDESC_BLOCK,x.block,x.rank);
        case REVDESC_CROSS_BLOCK:return lowdesc_pack(LOWDESC_CROSS,x.block,x.rank,x.depth);
        case REVDESC_CROSS_MAIN:
        default:std::cerr<<"cannot pack reverse LOW CROSS_MAIN\n";std::exit(264);
        }
    };
    for(size_t i=0;i<r.main_desc.size();++i)out.main_desc[i]=pack(r.main_desc[i]);
    for(size_t i=0;i<r.block_desc.size();++i)out.block_desc[i]=pack(r.block_desc[i]);
    return out;
}

static HighDescHost pack_reverse_high_descriptors(const ReverseHighDescHost&r){
    validate_reverse_high_cross_targets(r);
    HighDescHost out;
    out.main_base=r.main_base;out.block_base=r.block_base;
    out.main_total=r.main_total;out.block_total=r.block_total;
    out.main_observations=r.observations;out.main_cross=r.cross_main+r.cross_block;out.main_invalid=r.invalid;
    out.main_desc.resize(r.main_desc.size());out.block_desc.resize(r.block_desc.size());
    auto pack=[](const ReverseDesc&x)->uint32_t{
        switch(x.kind){
        case REVDESC_INVALID:return highdesc_pack(HIGHDESC_INVALID,0,0);
        case REVDESC_MAIN:return highdesc_pack(HIGHDESC_MAIN,x.block,x.rank);
        case REVDESC_BLOCK:return highdesc_pack(HIGHDESC_BLOCK,x.block,x.rank);
        case REVDESC_CROSS_MAIN:
        case REVDESC_CROSS_BLOCK:return highdesc_pack(HIGHDESC_CROSS,x.block,x.rank,x.depth);
        default:std::exit(267);
        }
    };
    for(size_t i=0;i<r.main_desc.size();++i)out.main_desc[i]=pack(r.main_desc[i]);
    for(size_t i=0;i<r.block_desc.size();++i)out.block_desc[i]=pack(r.block_desc[i]);
    return out;
}
