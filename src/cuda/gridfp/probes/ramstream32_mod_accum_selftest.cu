#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>

using Count = uint32_t;
#ifndef LOW_LUT_K
#define LOW_LUT_K 14
#endif
#ifndef HIGH_LUT_K
#define HIGH_LUT_K 13
#endif
__constant__ Count D_MOD;

#include "../ramstream32_mod_accum.cuh"

static bool check_one(uint64_t x,uint32_t mod){
    uint32_t got=gpu_direct_pm_reduce_u64_mod(x,mod);
    uint32_t want=mod?uint32_t(x%uint64_t(mod)):uint32_t(x);
    if(got!=want){
        std::cerr<<"FAIL pm-reduce x="<<x<<" mod="<<mod
                 <<" got="<<got<<" want="<<want<<'\n';
        return false;
    }
    return true;
}

int main(){
    std::array<uint32_t,12> mods={
        4294967295u, // c=1
        4294967291u, // first exact CRT prime, c=5
        4294967279u,
        4294966943u,
        4294966087u, // c=1209
        4294901761u, // c=65535, fast-path boundary
        4294901760u, // c=65536, generic fallback boundary
        2147483647u,
        1000000007u,
        65537u,
        3u,
        1u,
    };
    std::array<uint64_t,15> edges={
        0ull,1ull,2ull,3ull,
        0xfffffffeull,0xffffffffull,0x100000000ull,0x100000001ull,
        0x1ffffffffull,0xfffffffffffffffeull,0xffffffffffffffffull,
        uint64_t(GPU_DIRECT_PM_MAX_RAW_TERMS)*0xffffffffull,
        (1ull<<52)-1,(1ull<<52),(1ull<<53)-1,
    };
    for(uint32_t mod:mods){
        for(uint64_t x:edges)if(!check_one(x,mod))return 2;
        uint64_t m=mod;
        if(mod){
            std::array<uint64_t,8> around={m-1,m,m+1,2*m-1,2*m,2*m+1,3*m,3*m+1};
            for(uint64_t x:around)if(!check_one(x,mod))return 3;
        }
    }

    std::mt19937_64 rng(0x1618b300ULL);
    for(uint32_t mod:mods){
        for(int i=0;i<200000;++i){
            uint64_t x=rng();
            if(!check_one(x,mod))return 4;
        }
    }

    // The production closure accumulator never exceeds this bound even when
    // both 16-bit source counts are saturated and every CROSS op produces the
    // maximum possible number of inactive-half preimages.
    uint64_t worst=GPU_DIRECT_PM_MAX_RAW_TERMS*0xffffffffull;
    if(worst>=(1ull<<52))return 5;
    std::cout<<"pm-accum-selftest OK"
             <<" max_inverse_scan="<<GPU_DIRECT_PM_MAX_INVERSE_SCAN
             <<" max_raw_terms="<<GPU_DIRECT_PM_MAX_RAW_TERMS
             <<" worst_raw="<<worst
             <<" random_cases="<<(mods.size()*200000ull)<<'\n';
    return 0;
}
