#pragma once

#ifndef RP_RUNTIME_CHOOSE_SYM_U32_PREINCLUDE
#define RP_RUNTIME_CHOOSE_SYM_U32_PREINCLUDE 0
#endif
#ifndef RP_RUNTIME_CHOOSE_U32_PREINCLUDE_MODE
#define RP_RUNTIME_CHOOSE_U32_PREINCLUDE_MODE RP_RUNTIME_CHOOSE_SYM_U32_PREINCLUDE
#endif
static_assert(RP_RUNTIME_CHOOSE_U32_PREINCLUDE_MODE >= 0 &&
              RP_RUNTIME_CHOOSE_U32_PREINCLUDE_MODE <= 2,
              "RP_RUNTIME_CHOOSE_U32_PREINCLUDE_MODE must be 0, 1, or 2");

// Preserve the canonical host-upload symbol under a stable alternate name.
// Device-side indexing can then use either compact representation without
// changing cudaMemcpyToSymbol call sites.
#define RP_CHOOSE RP_CHOOSE_PREINCLUDE_ORIG
#include "gridfp_reduced_production_device.cuh"
#undef RP_CHOOSE

namespace oneesan::gridfp::reducedprod {

#if RP_RUNTIME_CHOOSE_U32_PREINCLUDE_MODE == 1
__device__ __constant__ std::uint32_t RP_RUNTIME_CHOOSE_SYM_U32[225] = {
#include "gridfp_reduced_production_choose_sym_u32_values.inc"
};
static_assert(sizeof(RP_RUNTIME_CHOOSE_SYM_U32) == 900);

__device__ __forceinline__ int runtime_choose_sym_u32_row_base_device(int n) {
    const int m = n >> 1;
    return (n & 1) ? (m + 1) * (m + 1) : m * (m + 1);
}

struct RuntimeChooseU32RowProxy {
    int n = 0;
    __device__ __forceinline__ Rank64 operator[](int k) const {
        if (n < 0 || n > RP_MAX_W || k < 0 || k > n) return 0;
        const int mirror = n - k;
        if (k > mirror) k = mirror;
        return RP_RUNTIME_CHOOSE_SYM_U32[
            runtime_choose_sym_u32_row_base_device(n) + k];
    }
};
#elif RP_RUNTIME_CHOOSE_U32_PREINCLUDE_MODE == 2
__device__ __constant__ std::uint32_t RP_RUNTIME_CHOOSE_TRI_U32[435] = {
#include "gridfp_reduced_production_choose_tri_u32_values.inc"
};
static_assert(sizeof(RP_RUNTIME_CHOOSE_TRI_U32) == 1740);

struct RuntimeChooseU32RowProxy {
    int n = 0;
    __device__ __forceinline__ Rank64 operator[](int k) const {
        if (n < 0 || n > RP_MAX_W || k < 0 || k > n) return 0;
        return RP_RUNTIME_CHOOSE_TRI_U32[n * (n + 1) / 2 + k];
    }
};
#endif

#if RP_RUNTIME_CHOOSE_U32_PREINCLUDE_MODE != 0
struct RuntimeChooseU32Proxy {
    __device__ __forceinline__ RuntimeChooseU32RowProxy operator[](int n) const {
        return RuntimeChooseU32RowProxy{n};
    }
};
#endif

} // namespace oneesan::gridfp::reducedprod

#if defined(__CUDA_ARCH__) && RP_RUNTIME_CHOOSE_U32_PREINCLUDE_MODE != 0
#define RP_CHOOSE RuntimeChooseU32Proxy{}
#else
#define RP_CHOOSE RP_CHOOSE_PREINCLUDE_ORIG
#endif
