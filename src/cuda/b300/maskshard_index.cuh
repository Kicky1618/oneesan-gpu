#pragma once

#include <cstdint>

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
