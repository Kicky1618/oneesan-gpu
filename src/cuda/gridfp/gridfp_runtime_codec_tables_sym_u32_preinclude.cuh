#pragma once

#ifndef RP_RUNTIME_CODEC_CHOOSE_SYM_U32
#define RP_RUNTIME_CODEC_CHOOSE_SYM_U32 0
#endif
#ifndef RP_RUNTIME_CODEC_PRIMITIVE_SYM_U32
#define RP_RUNTIME_CODEC_PRIMITIVE_SYM_U32 0
#endif
#ifndef RP_RUNTIME_CODEC_CHOOSE_U32_MODE
#define RP_RUNTIME_CODEC_CHOOSE_U32_MODE RP_RUNTIME_CODEC_CHOOSE_SYM_U32
#endif
#ifndef RP_RUNTIME_CODEC_PRIMITIVE_U32_MODE
#define RP_RUNTIME_CODEC_PRIMITIVE_U32_MODE RP_RUNTIME_CODEC_PRIMITIVE_SYM_U32
#endif
static_assert(RP_RUNTIME_CODEC_CHOOSE_U32_MODE >= 0 &&
              RP_RUNTIME_CODEC_CHOOSE_U32_MODE <= 3,
              "RP_RUNTIME_CODEC_CHOOSE_U32_MODE must be 0..3");
static_assert(RP_RUNTIME_CODEC_PRIMITIVE_U32_MODE >= 0 &&
              RP_RUNTIME_CODEC_PRIMITIVE_U32_MODE <= 2,
              "RP_RUNTIME_CODEC_PRIMITIVE_U32_MODE must be 0..2");

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

#if RP_RUNTIME_CODEC_CHOOSE_U32_MODE == 1
__device__ __constant__ std::uint32_t RP_RUNTIME_CODEC_CHOOSE_SYM_U32_TABLE[225] = {
#include "gridfp_reduced_production_choose_sym_u32_values.inc"
};
static_assert(sizeof(RP_RUNTIME_CODEC_CHOOSE_SYM_U32_TABLE) == 900);
struct RuntimeCodecChooseU32RowProxy {
    int n = 0;
    __device__ __forceinline__ Rank64 operator[](int k) const {
        if (n < 0 || n > RP_MAX_W || k < 0 || k > n) return 0;
        const int mirror = n - k;
        if (k > mirror) k = mirror;
        return RP_RUNTIME_CODEC_CHOOSE_SYM_U32_TABLE[
            runtime_codec_sym_u32_row_base_device(n) + k];
    }
};
#elif RP_RUNTIME_CODEC_CHOOSE_U32_MODE == 2
__device__ __constant__ std::uint32_t RP_RUNTIME_CODEC_CHOOSE_TRI_U32_TABLE[435] = {
#include "gridfp_reduced_production_choose_tri_u32_values.inc"
};
static_assert(sizeof(RP_RUNTIME_CODEC_CHOOSE_TRI_U32_TABLE) == 1740);
struct RuntimeCodecChooseU32RowProxy {
    int n = 0;
    __device__ __forceinline__ Rank64 operator[](int k) const {
        if (n < 0 || n > RP_MAX_W || k < 0 || k > n) return 0;
        return RP_RUNTIME_CODEC_CHOOSE_TRI_U32_TABLE[n * (n + 1) / 2 + k];
    }
};
#elif RP_RUNTIME_CODEC_CHOOSE_U32_MODE == 3
__device__ __constant__ std::uint32_t
RP_RUNTIME_CODEC_CHOOSE_FULL_U32_TABLE[RP_MAX_W + 1][RP_MAX_W + 1] = {
#include "gridfp_reduced_production_choose_full_u32_values.inc"
};
static_assert(sizeof(RP_RUNTIME_CODEC_CHOOSE_FULL_U32_TABLE) == 3364);
struct RuntimeCodecChooseU32RowProxy {
    int n = 0;
    __device__ __forceinline__ Rank64 operator[](int k) const {
        if (n < 0 || n > RP_MAX_W || k < 0 || k > n) return 0;
        return RP_RUNTIME_CODEC_CHOOSE_FULL_U32_TABLE[n][k];
    }
};
#endif

#if RP_RUNTIME_CODEC_CHOOSE_U32_MODE != 0
struct RuntimeCodecChooseU32Proxy {
    __device__ __forceinline__ RuntimeCodecChooseU32RowProxy operator[](int n) const {
        return RuntimeCodecChooseU32RowProxy{n};
    }
};
#endif

#if RP_RUNTIME_CODEC_PRIMITIVE_U32_MODE == 1
__device__ __constant__ std::uint32_t RP_RUNTIME_CODEC_PRIMITIVE_SYM_U32_TABLE[225] = {
#include "gridfp_reduced_production_primitive_sym_u32_values.inc"
};
static_assert(sizeof(RP_RUNTIME_CODEC_PRIMITIVE_SYM_U32_TABLE) == 900);
struct RuntimeCodecPrimitiveU32RowProxy {
    int rem = 0;
    __device__ __forceinline__ Rank64 operator[](int h) const {
        if (rem < 0 || rem > RP_MAX_W || h < 0 || h > rem || ((rem ^ h) & 1))
            return 0;
        return RP_RUNTIME_CODEC_PRIMITIVE_SYM_U32_TABLE[
            runtime_codec_sym_u32_row_base_device(rem) + (h >> 1)];
    }
};
#elif RP_RUNTIME_CODEC_PRIMITIVE_U32_MODE == 2
__device__ __constant__ std::uint32_t
RP_RUNTIME_CODEC_PRIMITIVE_FULL_U32_TABLE[RP_MAX_W + 1][RP_MAX_W + 2] = {
#include "gridfp_reduced_production_primitive_full_u32_values_0_9.inc"
,
#include "gridfp_reduced_production_primitive_full_u32_values_10_19.inc"
,
#include "gridfp_reduced_production_primitive_full_u32_values_20_28.inc"
};
static_assert(sizeof(RP_RUNTIME_CODEC_PRIMITIVE_FULL_U32_TABLE) == 3480);
struct RuntimeCodecPrimitiveU32RowProxy {
    int rem = 0;
    __device__ __forceinline__ Rank64 operator[](int h) const {
        if (rem < 0 || rem > RP_MAX_W || h < 0 || h > RP_MAX_W + 1) return 0;
        return RP_RUNTIME_CODEC_PRIMITIVE_FULL_U32_TABLE[rem][h];
    }
};
#endif

#if RP_RUNTIME_CODEC_PRIMITIVE_U32_MODE != 0
struct RuntimeCodecPrimitiveU32Proxy {
    __device__ __forceinline__ RuntimeCodecPrimitiveU32RowProxy operator[](int rem) const {
        return RuntimeCodecPrimitiveU32RowProxy{rem};
    }
};
#endif

} // namespace oneesan::gridfp::reducedprod

#if defined(__CUDA_ARCH__) && RP_RUNTIME_CODEC_CHOOSE_U32_MODE != 0
#define RP_CHOOSE RuntimeCodecChooseU32Proxy{}
#else
#define RP_CHOOSE RP_CHOOSE_CODEC_TABLES_ORIG
#endif

#if defined(__CUDA_ARCH__) && RP_RUNTIME_CODEC_PRIMITIVE_U32_MODE != 0
#define RP_PRIMITIVE RuntimeCodecPrimitiveU32Proxy{}
#else
#define RP_PRIMITIVE RP_PRIMITIVE_CODEC_TABLES_ORIG
#endif
