#pragma once

#ifndef RP_RUNTIME_CHOOSE_SYM_U32_PREINCLUDE
#define RP_RUNTIME_CHOOSE_SYM_U32_PREINCLUDE 0
#endif
static_assert(RP_RUNTIME_CHOOSE_SYM_U32_PREINCLUDE == 0 ||
              RP_RUNTIME_CHOOSE_SYM_U32_PREINCLUDE == 1,
              "RP_RUNTIME_CHOOSE_SYM_U32_PREINCLUDE must be 0 or 1");

// Preserve the canonical host-upload symbol under a stable alternate name.
// Later device-side RP_CHOOSE indexing can then be redirected to the compact
// symmetric uint32 table without changing cudaMemcpyToSymbol call sites.
#define RP_CHOOSE RP_CHOOSE_PREINCLUDE_ORIG
#include "gridfp_reduced_production_device.cuh"
#undef RP_CHOOSE

namespace oneesan::gridfp::reducedprod {

#if RP_RUNTIME_CHOOSE_SYM_U32_PREINCLUDE
static constexpr int RP_RUNTIME_CHOOSE_SYM_U32_ENTRIES = 225;
__device__ __constant__ std::uint32_t
RP_RUNTIME_CHOOSE_SYM_U32[RP_RUNTIME_CHOOSE_SYM_U32_ENTRIES] = {
#include "gridfp_reduced_production_choose_sym_u32_values.inc"
};
static_assert(sizeof(RP_RUNTIME_CHOOSE_SYM_U32) == 900);

__device__ __forceinline__ int runtime_choose_sym_u32_row_base_device(int n) {
    const int m = n >> 1;
    return (n & 1) ? (m + 1) * (m + 1) : m * (m + 1);
}

struct RuntimeChooseSymU32RowProxy {
    int n = 0;
    __device__ __forceinline__ Rank64 operator[](int k) const {
        if (n < 0 || n > RP_MAX_W || k < 0 || k > n) return 0;
        const int mirror = n - k;
        if (k > mirror) k = mirror;
        return RP_RUNTIME_CHOOSE_SYM_U32[
            runtime_choose_sym_u32_row_base_device(n) + k];
    }
};

struct RuntimeChooseSymU32Proxy {
    __device__ __forceinline__ RuntimeChooseSymU32RowProxy operator[](int n) const {
        return RuntimeChooseSymU32RowProxy{n};
    }
};
#endif

} // namespace oneesan::gridfp::reducedprod

#if defined(__CUDA_ARCH__) && RP_RUNTIME_CHOOSE_SYM_U32_PREINCLUDE
#define RP_CHOOSE RuntimeChooseSymU32Proxy{}
#else
#define RP_CHOOSE RP_CHOOSE_PREINCLUDE_ORIG
#endif
