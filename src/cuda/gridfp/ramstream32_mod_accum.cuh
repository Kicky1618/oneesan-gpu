#pragma once

#include <cstdint>

#ifndef GPU_DIRECT_PM_ACCUM
#define GPU_DIRECT_PM_ACCUM 0
#endif

#if defined(__CUDACC__)
#define GDMA_HD __host__ __device__ __forceinline__
#else
#define GDMA_HD inline
#endif

// Reduce x modulo mod. The fast path exploits mod = B - c, B = 2^32.
// One fold F(x)=lo32(x)+hi32(x)*c preserves x modulo mod.
// For uint64_t x and c<=65535:
//   x1 < B(c+1), so hi(x1)<=c;
//   x2 <= (B-1)+c^2 < 2B, so hi(x2)<=1;
//   x3 < B+c;
//   x4 < B.
// Therefore four folds make truncation to uint32_t exact, and because x4<B
// while mod=B-c, at most one final subtraction is needed. Exact CRT primes in
// this repository have c<=1209. Other moduli use the generic fallback.
GDMA_HD uint32_t gpu_direct_pm_reduce_u64_mod(uint64_t x, uint32_t mod) {
    if (!mod) return uint32_t(x);
    const uint32_t c = uint32_t(0u - mod); // 2^32 - mod, modulo 2^32.
    if (c <= 65535u) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            x = uint64_t(uint32_t(x)) + uint64_t(uint32_t(x >> 32)) * c;
        }
        uint32_t r = uint32_t(x);
        if (r >= mod) r -= mod;
        return r;
    }
    return uint32_t(x % uint64_t(mod));
}

#undef GDMA_HD

#if defined(__CUDACC__)
__device__ __forceinline__ Count gpu_direct_pm_reduce_u64(uint64_t x) {
    return Count(gpu_direct_pm_reduce_u64_mod(x, uint32_t(D_MOD)));
}
#endif

static constexpr uint64_t GPU_DIRECT_PM_MAX_INVERSE_SCAN =
    uint64_t(LOW_LUT_K > HIGH_LUT_K ? LOW_LUT_K : HIGH_LUT_K);
static constexpr uint64_t GPU_DIRECT_PM_MAX_RAW_TERMS =
    1ull + 0xffffull + 0xffffull * GPU_DIRECT_PM_MAX_INVERSE_SCAN;
static_assert(GPU_DIRECT_PM_MAX_RAW_TERMS < (1ull << 20),
              "raw closure accumulator term bound unexpectedly large");
static_assert(GPU_DIRECT_PM_MAX_RAW_TERMS * 0xffffffffull < (1ull << 52),
              "raw closure accumulator may overflow uint64_t safety budget");
