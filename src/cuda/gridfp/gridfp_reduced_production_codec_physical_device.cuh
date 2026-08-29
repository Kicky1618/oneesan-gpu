#pragma once

#ifndef RP_EXPERIMENTAL_CODEC_CHOOSE_PHYSICAL_MODE
#define RP_EXPERIMENTAL_CODEC_CHOOSE_PHYSICAL_MODE 0
#endif
#ifndef RP_EXPERIMENTAL_CODEC_PRIMITIVE_PHYSICAL_MODE
#define RP_EXPERIMENTAL_CODEC_PRIMITIVE_PHYSICAL_MODE 0
#endif

static_assert(RP_EXPERIMENTAL_CODEC_CHOOSE_PHYSICAL_MODE >= 0 &&
              RP_EXPERIMENTAL_CODEC_CHOOSE_PHYSICAL_MODE <= 3,
              "RP_EXPERIMENTAL_CODEC_CHOOSE_PHYSICAL_MODE must be 0..3");
static_assert(RP_EXPERIMENTAL_CODEC_PRIMITIVE_PHYSICAL_MODE >= 0 &&
              RP_EXPERIMENTAL_CODEC_PRIMITIVE_PHYSICAL_MODE <= 2,
              "RP_EXPERIMENTAL_CODEC_PRIMITIVE_PHYSICAL_MODE must be 0..2");

#if RP_EXPERIMENTAL_CODEC_CHOOSE_PHYSICAL_MODE != 0 && defined(RP_CHOOSE)
#error "physical choose layout cannot be combined with an RP_CHOOSE preinclude remap"
#endif
#if RP_EXPERIMENTAL_CODEC_PRIMITIVE_PHYSICAL_MODE != 0 && defined(RP_PRIMITIVE)
#error "physical primitive layout cannot be combined with an RP_PRIMITIVE preinclude remap"
#endif

__device__ __forceinline__ int codec_physical_sym_row_base_device(int n) {
    const int m = n >> 1;
    return (n & 1) ? (m + 1) * (m + 1) : m * (m + 1);
}

#if RP_EXPERIMENTAL_CODEC_CHOOSE_PHYSICAL_MODE == 0
__constant__ Rank64 RP_CHOOSE[RP_MAX_W + 1][RP_MAX_W + 1];
static constexpr int RP_CODEC_PHYSICAL_CHOOSE_BYTES =
    (RP_MAX_W + 1) * (RP_MAX_W + 1) * int(sizeof(Rank64));
static constexpr int RP_CODEC_PHYSICAL_CHOOSE_UPLOAD_SINK_BYTES = 0;
#elif RP_EXPERIMENTAL_CODEC_CHOOSE_PHYSICAL_MODE == 1
// Preserve the existing host cudaMemcpyToSymbol call without retaining the
// legacy table in constant memory. Device code is redirected to the u32 table.
__device__ Rank64 RP_CHOOSE[RP_MAX_W + 1][RP_MAX_W + 1];
__device__ __constant__ std::uint32_t RP_CODEC_PHYSICAL_CHOOSE_SYM_U32[225] = {
#include "gridfp_reduced_production_choose_sym_u32_values.inc"
};
static_assert(sizeof(RP_CODEC_PHYSICAL_CHOOSE_SYM_U32) == 900);
static constexpr int RP_CODEC_PHYSICAL_CHOOSE_BYTES = 900;
static constexpr int RP_CODEC_PHYSICAL_CHOOSE_UPLOAD_SINK_BYTES = 6728;
struct CodecPhysicalChooseRowProxy {
    int n = 0;
    __device__ __forceinline__ Rank64 operator[](int k) const {
        if (n < 0 || n > RP_MAX_W || k < 0 || k > n) return 0;
        const int mirror = n - k;
        if (k > mirror) k = mirror;
        return RP_CODEC_PHYSICAL_CHOOSE_SYM_U32[
            codec_physical_sym_row_base_device(n) + k];
    }
};
#elif RP_EXPERIMENTAL_CODEC_CHOOSE_PHYSICAL_MODE == 2
__device__ Rank64 RP_CHOOSE[RP_MAX_W + 1][RP_MAX_W + 1];
__device__ __constant__ std::uint32_t RP_CODEC_PHYSICAL_CHOOSE_TRI_U32[435] = {
#include "gridfp_reduced_production_choose_tri_u32_values.inc"
};
static_assert(sizeof(RP_CODEC_PHYSICAL_CHOOSE_TRI_U32) == 1740);
static constexpr int RP_CODEC_PHYSICAL_CHOOSE_BYTES = 1740;
static constexpr int RP_CODEC_PHYSICAL_CHOOSE_UPLOAD_SINK_BYTES = 6728;
struct CodecPhysicalChooseRowProxy {
    int n = 0;
    __device__ __forceinline__ Rank64 operator[](int k) const {
        if (n < 0 || n > RP_MAX_W || k < 0 || k > n) return 0;
        return RP_CODEC_PHYSICAL_CHOOSE_TRI_U32[n * (n + 1) / 2 + k];
    }
};
#else
__device__ Rank64 RP_CHOOSE[RP_MAX_W + 1][RP_MAX_W + 1];
__device__ __constant__ std::uint32_t
RP_CODEC_PHYSICAL_CHOOSE_FULL_U32[RP_MAX_W + 1][RP_MAX_W + 1] = {
#include "gridfp_reduced_production_choose_full_u32_values.inc"
};
static_assert(sizeof(RP_CODEC_PHYSICAL_CHOOSE_FULL_U32) == 3364);
static constexpr int RP_CODEC_PHYSICAL_CHOOSE_BYTES = 3364;
static constexpr int RP_CODEC_PHYSICAL_CHOOSE_UPLOAD_SINK_BYTES = 6728;
struct CodecPhysicalChooseRowProxy {
    int n = 0;
    __device__ __forceinline__ Rank64 operator[](int k) const {
        if (n < 0 || n > RP_MAX_W || k < 0 || k > RP_MAX_W) return 0;
        return RP_CODEC_PHYSICAL_CHOOSE_FULL_U32[n][k];
    }
};
#endif

#if RP_EXPERIMENTAL_CODEC_CHOOSE_PHYSICAL_MODE != 0
struct CodecPhysicalChooseProxy {
    __device__ __forceinline__ CodecPhysicalChooseRowProxy operator[](int n) const {
        return CodecPhysicalChooseRowProxy{n};
    }
};
#if defined(__CUDA_ARCH__)
#define RP_CHOOSE CodecPhysicalChooseProxy{}
#endif
#endif

#if RP_EXPERIMENTAL_CODEC_PRIMITIVE_PHYSICAL_MODE == 0
__constant__ Rank64 RP_PRIMITIVE[RP_MAX_W + 1][RP_MAX_W + 2];
static constexpr int RP_CODEC_PHYSICAL_PRIMITIVE_BYTES =
    (RP_MAX_W + 1) * (RP_MAX_W + 2) * int(sizeof(Rank64));
static constexpr int RP_CODEC_PHYSICAL_PRIMITIVE_UPLOAD_SINK_BYTES = 0;
#elif RP_EXPERIMENTAL_CODEC_PRIMITIVE_PHYSICAL_MODE == 1
__device__ Rank64 RP_PRIMITIVE[RP_MAX_W + 1][RP_MAX_W + 2];
__device__ __constant__ std::uint32_t RP_CODEC_PHYSICAL_PRIMITIVE_SYM_U32[225] = {
#include "gridfp_reduced_production_primitive_sym_u32_values.inc"
};
static_assert(sizeof(RP_CODEC_PHYSICAL_PRIMITIVE_SYM_U32) == 900);
static constexpr int RP_CODEC_PHYSICAL_PRIMITIVE_BYTES = 900;
static constexpr int RP_CODEC_PHYSICAL_PRIMITIVE_UPLOAD_SINK_BYTES = 6960;
struct CodecPhysicalPrimitiveRowProxy {
    int rem = 0;
    __device__ __forceinline__ Rank64 operator[](int h) const {
        if (rem < 0 || rem > RP_MAX_W || h < 0 || h > rem || ((rem ^ h) & 1))
            return 0;
        return RP_CODEC_PHYSICAL_PRIMITIVE_SYM_U32[
            codec_physical_sym_row_base_device(rem) + (h >> 1)];
    }
};
#else
__device__ Rank64 RP_PRIMITIVE[RP_MAX_W + 1][RP_MAX_W + 2];
__device__ __constant__ std::uint32_t
RP_CODEC_PHYSICAL_PRIMITIVE_FULL_U32[RP_MAX_W + 1][RP_MAX_W + 2] = {
#include "gridfp_reduced_production_primitive_full_u32_values_0_9.inc"
,
#include "gridfp_reduced_production_primitive_full_u32_values_10_19.inc"
,
#include "gridfp_reduced_production_primitive_full_u32_values_20_28.inc"
};
static_assert(sizeof(RP_CODEC_PHYSICAL_PRIMITIVE_FULL_U32) == 3480);
static constexpr int RP_CODEC_PHYSICAL_PRIMITIVE_BYTES = 3480;
static constexpr int RP_CODEC_PHYSICAL_PRIMITIVE_UPLOAD_SINK_BYTES = 6960;
struct CodecPhysicalPrimitiveRowProxy {
    int rem = 0;
    __device__ __forceinline__ Rank64 operator[](int h) const {
        if (rem < 0 || rem > RP_MAX_W || h < 0 || h > RP_MAX_W + 1) return 0;
        return RP_CODEC_PHYSICAL_PRIMITIVE_FULL_U32[rem][h];
    }
};
#endif

#if RP_EXPERIMENTAL_CODEC_PRIMITIVE_PHYSICAL_MODE != 0
struct CodecPhysicalPrimitiveProxy {
    __device__ __forceinline__ CodecPhysicalPrimitiveRowProxy operator[](int rem) const {
        return CodecPhysicalPrimitiveRowProxy{rem};
    }
};
#if defined(__CUDA_ARCH__)
#define RP_PRIMITIVE CodecPhysicalPrimitiveProxy{}
#endif
#endif

static constexpr int RP_CODEC_PHYSICAL_TABLE_BYTES =
    RP_CODEC_PHYSICAL_CHOOSE_BYTES + RP_CODEC_PHYSICAL_PRIMITIVE_BYTES;
static constexpr int RP_CODEC_PHYSICAL_UPLOAD_SINK_BYTES =
    RP_CODEC_PHYSICAL_CHOOSE_UPLOAD_SINK_BYTES +
    RP_CODEC_PHYSICAL_PRIMITIVE_UPLOAD_SINK_BYTES;
