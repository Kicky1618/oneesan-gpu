#pragma once

#include "gridfp_reduced_production_device.cuh"

namespace oneesan::gridfp::reducedprod {

#ifndef RP_RUNTIME_FAST_DIV64
#define RP_RUNTIME_FAST_DIV64 1
#endif
static_assert(RP_RUNTIME_FAST_DIV64 == 0 || RP_RUNTIME_FAST_DIV64 == 1,
              "RP_RUNTIME_FAST_DIV64 must be 0 or 1");

// For d>1, magic=ceil(2^64/d). Then floor(x*magic/2^64) is either
// floor(x/d) or exactly one larger. Detect the latter without division by
// examining the high word of q0*d and its low word versus x.
__device__ __forceinline__ void runtime_fastdivmod64_magic(
    Rank64 x,
    Rank64 divisor,
    Rank64 magic,
    Rank64& quotient,
    Rank64& remainder
) {
#if RP_RUNTIME_FAST_DIV64
    if (divisor == 1) {
        quotient = x;
        remainder = 0;
        return;
    }
    Rank64 q = __umul64hi(x, magic);
    const Rank64 product_lo = q * divisor;
    const Rank64 product_hi = __umul64hi(q, divisor);
    if (product_hi || product_lo > x) --q;
    quotient = q;
    remainder = x - q * divisor;
#else
    quotient = x / divisor;
    remainder = x % divisor;
#endif
}

} // namespace oneesan::gridfp::reducedprod
