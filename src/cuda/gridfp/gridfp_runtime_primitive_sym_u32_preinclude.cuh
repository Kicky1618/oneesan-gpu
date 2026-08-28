#pragma once

#ifndef RP_RUNTIME_PRIMITIVE_SYM_U32_PREINCLUDE
#define RP_RUNTIME_PRIMITIVE_SYM_U32_PREINCLUDE 0
#endif
static_assert(RP_RUNTIME_PRIMITIVE_SYM_U32_PREINCLUDE == 0 ||
              RP_RUNTIME_PRIMITIVE_SYM_U32_PREINCLUDE == 1,
              "RP_RUNTIME_PRIMITIVE_SYM_U32_PREINCLUDE must be 0 or 1");

// Keep the canonical host-upload symbol under an alternate name, then redirect
// device-side indexing to the parity-sparse uint32 representation.
#define RP_PRIMITIVE RP_PRIMITIVE_PREINCLUDE_ORIG
#include "gridfp_reduced_production_device.cuh"
#undef RP_PRIMITIVE

namespace oneesan::gridfp::reducedprod {

#if RP_RUNTIME_PRIMITIVE_SYM_U32_PREINCLUDE
static constexpr int RP_RUNTIME_PRIMITIVE_SYM_U32_ENTRIES = 225;
__device__ __constant__ std::uint32_t
RP_RUNTIME_PRIMITIVE_SYM_U32[RP_RUNTIME_PRIMITIVE_SYM_U32_ENTRIES] = {
#include "gridfp_reduced_production_primitive_sym_u32_values.inc"
};
static_assert(sizeof(RP_RUNTIME_PRIMITIVE_SYM_U32) == 900);

__device__ __forceinline__ int runtime_primitive_sym_u32_row_base_device(int rem) {
    const int m = rem >> 1;
    return (rem & 1) ? (m + 1) * (m + 1) : m * (m + 1);
}

struct RuntimePrimitiveSymU32RowProxy {
    int rem = 0;
    __device__ __forceinline__ Rank64 operator[](int h) const {
        if (rem < 0 || rem > RP_MAX_W || h < 0 || h > rem || ((rem ^ h) & 1))
            return 0;
        return RP_RUNTIME_PRIMITIVE_SYM_U32[
            runtime_primitive_sym_u32_row_base_device(rem) + (h >> 1)];
    }
};

struct RuntimePrimitiveSymU32Proxy {
    __device__ __forceinline__ RuntimePrimitiveSymU32RowProxy operator[](int rem) const {
        return RuntimePrimitiveSymU32RowProxy{rem};
    }
};
#endif

} // namespace oneesan::gridfp::reducedprod

#if defined(__CUDA_ARCH__) && RP_RUNTIME_PRIMITIVE_SYM_U32_PREINCLUDE
#define RP_PRIMITIVE RuntimePrimitiveSymU32Proxy{}
#else
#define RP_PRIMITIVE RP_PRIMITIVE_PREINCLUDE_ORIG
#endif
