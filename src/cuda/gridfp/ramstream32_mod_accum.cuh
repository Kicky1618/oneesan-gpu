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

// Reduce x modulo mod. The fast path exploits mod = 2^32 - c.  For c <=
// 65535, three base-2^32 folds reduce any uint64_t to < 2*mod, so only one
// final subtraction is required.  Exact CRT primes used by this repository
// have c <= 1209.  The generic fallback keeps experimental/smoke moduli valid.
GDMA_HD uint32_t gpu_direct_pm_reduce_u64_mod(uint64_t x, uint32_t mod) {
    if (!mod) return uint32_t(x);
    const uint32_t c = uint32_t(0u - mod); // 2^32 - mod, modulo 2^32.
    if (c <= 65535u) {
#pragma unroll
        for (int i = 0; i < 3; ++i) {
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

static constexpr uint64_t GPU_DIRECT_PM_MAX_RAW_TERMS =
    1ull + 0xffffull + 0xffffull * uint64_t(MAXW);
static_assert(GPU_DIRECT_PM_MAX_RAW_TERMS < (1ull << 20),
              "raw closure accumulator term bound unexpectedly large");
static_assert(GPU_DIRECT_PM_MAX_RAW_TERMS * 0xffffffffull < (1ull << 52),
              "raw closure accumulator may overflow uint64_t safety budget");
