#pragma once

#include <cstdint>

// HIGH closure row-depth pruning is parsed before the v0.15 row-depth headers
// in the shared batch include order. Forward declarations are sufficient; the
// actual constant-memory definitions are provided later by
// maskshard_rowdepth_fblock_io.cuh / maskshard_rowdepth_exact_io.cuh.
#ifdef MASKSHARD_HIGH_CLOSURE_ROW_DEPTH
extern __device__ __constant__ int D_MS_ROW_DEPTH_INDEX;
extern __device__ __constant__ std::uint8_t* D_MS_ROW_DEPTH_LOW_PEAK;
extern __device__ __constant__ std::uint8_t* D_MS_ROW_DEPTH_HIGH_PEAK;
#endif

// Split one factorized group rank into (row, column). Production n=27 mask
// groups are below 2^32 elements, so the hot path uses 32-bit quotient and
// remainder. Keep the generic 64-bit fallback for other experimental widths or
// splits rather than relying on the current group-size bound for correctness.
__device__ __forceinline__ void maskshard_split_rank(
    Code i, const FBlock& x, uint32_t& row, uint32_t& col
) {
    const Code r64 = i - x.off;
    if (r64 <= 0xffffffffULL) {
        const uint32_t r = uint32_t(r64);
        row = x.stride ? r / x.stride : 0;
        col = x.stride ? r - row * x.stride : 0;
    } else {
        row = x.stride ? uint32_t(r64 / x.stride) : 0;
        col = x.stride ? uint32_t(r64 - Code(row) * x.stride) : 0;
    }
}
