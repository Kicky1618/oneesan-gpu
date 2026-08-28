#pragma once

#ifndef RP_RUNTIME_CODEC_CHOOSE_SYM_U32
#define RP_RUNTIME_CODEC_CHOOSE_SYM_U32 0
#endif
#ifndef RP_RUNTIME_CODEC_PRIMITIVE_SYM_U32
#define RP_RUNTIME_CODEC_PRIMITIVE_SYM_U32 0
#endif
static_assert(RP_RUNTIME_CODEC_CHOOSE_SYM_U32 == 0 ||
              RP_RUNTIME_CODEC_CHOOSE_SYM_U32 == 1,
              "RP_RUNTIME_CODEC_CHOOSE_SYM_U32 must be 0 or 1");
static_assert(RP_RUNTIME_CODEC_PRIMITIVE_SYM_U32 == 0 ||
              RP_RUNTIME_CODEC_PRIMITIVE_SYM_U32 == 1,
              "RP_RUNTIME_CODEC_PRIMITIVE_SYM_U32 must be 0 or 1");

#define RP_CHOOSE RP_CHOOSE_CODEC_TABLES_ORIG
#define RP_PRIMITIVE RP_PRIMITIVE_CODEC_TABLES_ORIG
#include "gridfp_reduced_production_device.cuh"
#undef RP_PRIMITIVE
#undef RP_CHOOSE

namespace oneesan::gridfp::reducedprod {

__device__ __forceinline__ int runtime_codec_sym_u32_row_base_device(int n) {
    const int m = n >> 1;
    return (n & 1) ? (m + 1) * (m + 1) : m * (m + 1);
}

#if RP_RUNTIME_CODEC_CHOOSE_SYM_U32
__device__ __constant__ std::uint32_t RP_RUNTIME_CODEC_CHOOSE_SYM_U32_TABLE[225] = {
#include "gridfp_reduced_production_choose_sym_u32_values.inc"
};
static_assert(sizeof(RP_RUNTIME_CODEC_CHOOSE_SYM_U32_TABLE) == 900);

struct RuntimeCodecChooseSymU32RowProxy {
    int n = 0;
    __device__ __forceinline__ Rank64 operator[](int k) const {
        if (n < 0 || n > RP_MAX_W || k < 0 || k > n) return 0;
        const int mirror = n - k;
        if (k > mirror) k = mirror;
        return RP_RUNTIME_CODEC_CHOOSE_SYM_U32_TABLE[
            runtime_codec_sym_u32_row_base_device(n) + k];
    }
};
struct RuntimeCodecChooseSymU32Proxy {
    __device__ __forceinline__ RuntimeCodecChooseSymU32RowProxy operator[](int n) const {
        return RuntimeCodecChooseSymU32RowProxy{n};
    }
};
#endif

#if RP_RUNTIME_CODEC_PRIMITIVE_SYM_U32
__device__ __constant__ std::uint32_t RP_RUNTIME_CODEC_PRIMITIVE_SYM_U32_TABLE[225] = {
#include "gridfp_reduced_production_primitive_sym_u32_values.inc"
};
static_assert(sizeof(RP_RUNTIME_CODEC_PRIMITIVE_SYM_U32_TABLE) == 900);

struct RuntimeCodecPrimitiveSymU32RowProxy {
    int rem = 0;
    __device__ __forceinline__ Rank64 operator[](int h) const {
        if (rem < 0 || rem > RP_MAX_W || h < 0 || h > rem || ((rem ^ h) & 1))
            return 0;
        return RP_RUNTIME_CODEC_PRIMITIVE_SYM_U32_TABLE[
            runtime_codec_sym_u32_row_base_device(rem) + (h >> 1)];
    }
};
struct RuntimeCodecPrimitiveSymU32Proxy {
    __device__ __forceinline__ RuntimeCodecPrimitiveSymU32RowProxy operator[](int rem) const {
        return RuntimeCodecPrimitiveSymU32RowProxy{rem};
    }
};
#endif

} // namespace oneesan::gridfp::reducedprod

#if defined(__CUDA_ARCH__) && RP_RUNTIME_CODEC_CHOOSE_SYM_U32
#define RP_CHOOSE RuntimeCodecChooseSymU32Proxy{}
#else
#define RP_CHOOSE RP_CHOOSE_CODEC_TABLES_ORIG
#endif

#if defined(__CUDA_ARCH__) && RP_RUNTIME_CODEC_PRIMITIVE_SYM_U32
#define RP_PRIMITIVE RuntimeCodecPrimitiveSymU32Proxy{}
#else
#define RP_PRIMITIVE RP_PRIMITIVE_CODEC_TABLES_ORIG
#endif
