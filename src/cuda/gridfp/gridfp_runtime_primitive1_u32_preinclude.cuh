#pragma once

#ifndef RP_RUNTIME_PRIMITIVE1_SECTOR_PREINCLUDE
#define RP_RUNTIME_PRIMITIVE1_SECTOR_PREINCLUDE 0
#endif
#ifndef RP_RUNTIME_PRIMITIVE1_U32_PREINCLUDE
#define RP_RUNTIME_PRIMITIVE1_U32_PREINCLUDE 0
#endif
static_assert(RP_RUNTIME_PRIMITIVE1_SECTOR_PREINCLUDE == 0 ||
              RP_RUNTIME_PRIMITIVE1_SECTOR_PREINCLUDE == 1,
              "RP_RUNTIME_PRIMITIVE1_SECTOR_PREINCLUDE must be 0 or 1");
static_assert(RP_RUNTIME_PRIMITIVE1_U32_PREINCLUDE == 0 ||
              RP_RUNTIME_PRIMITIVE1_U32_PREINCLUDE == 1,
              "RP_RUNTIME_PRIMITIVE1_U32_PREINCLUDE must be 0 or 1");
static_assert(!(RP_RUNTIME_PRIMITIVE1_SECTOR_PREINCLUDE &&
                RP_RUNTIME_PRIMITIVE1_U32_PREINCLUDE),
              "primitive1 sector and u32 preinclude modes are mutually exclusive");

// Force the canonical primitive DP symbol to a stable alternate name while
// parsing the base codec. Host-side table upload code later in the translation
// unit is remapped back to the same alternate symbol, so cudaMemcpyToSymbol
// remains valid. Only subsequent device-side direct RP_PRIMITIVE indexing is
// optionally intercepted.
#define RP_PRIMITIVE RP_PRIMITIVE_PREINCLUDE_ORIG
#include "gridfp_reduced_production_codec_device.cuh"
#undef RP_PRIMITIVE

namespace oneesan::gridfp::reducedprod {

#if RP_RUNTIME_PRIMITIVE1_U32_PREINCLUDE
__device__ __constant__ std::uint32_t RP_RUNTIME_PRIMITIVE1_U32[14] = {
    1u, 2u, 5u, 14u, 42u, 132u, 429u, 1430u,
    4862u, 16796u, 58786u, 208012u, 742900u, 2674440u};
static_assert(sizeof(RP_RUNTIME_PRIMITIVE1_U32) == 56);
#endif

#if RP_RUNTIME_PRIMITIVE1_SECTOR_PREINCLUDE || RP_RUNTIME_PRIMITIVE1_U32_PREINCLUDE
struct RuntimePrimitive1RowProxy {
    int rem = 0;
    __device__ __forceinline__ Rank64 operator[](int j) const {
        if (j == 1 && rem >= 1 && rem <= 27 && (rem & 1)) {
#if RP_RUNTIME_PRIMITIVE1_U32_PREINCLUDE
            return RP_RUNTIME_PRIMITIVE1_U32[rem >> 1];
#else
            return RP_SECTOR_PRIMITIVE[rem >> 1];
#endif
        }
        return RP_PRIMITIVE_PREINCLUDE_ORIG[rem][j];
    }
};

struct RuntimePrimitive1Proxy {
    __device__ __forceinline__ RuntimePrimitive1RowProxy operator[](int rem) const {
        return RuntimePrimitive1RowProxy{rem};
    }
};
#endif

} // namespace oneesan::gridfp::reducedprod

#if defined(__CUDA_ARCH__) && \
    (RP_RUNTIME_PRIMITIVE1_SECTOR_PREINCLUDE || RP_RUNTIME_PRIMITIVE1_U32_PREINCLUDE)
#define RP_PRIMITIVE RuntimePrimitive1Proxy{}
#else
#define RP_PRIMITIVE RP_PRIMITIVE_PREINCLUDE_ORIG
#endif
