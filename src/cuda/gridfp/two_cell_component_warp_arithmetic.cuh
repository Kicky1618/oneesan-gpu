#pragma once

#include "../../common/two_cell_component_matching.cuh"

#include <cuda_runtime.h>

namespace oneesan::twocell::cuda_component {

constexpr unsigned kFullWarp = 0xffffffffu;

__device__ __forceinline__ std::uint32_t add_mod(
    std::uint32_t a,
    std::uint32_t b,
    std::uint32_t mod
) {
    const unsigned long long z =
        static_cast<unsigned long long>(a) + static_cast<unsigned long long>(b);
    return static_cast<std::uint32_t>(z >= mod ? z - mod : z);
}

__device__ __forceinline__ std::uint32_t warp_tail_sum_mod(
    std::uint32_t x,
    int lane,
    int n,
    std::uint32_t mod
) {
    std::uint32_t z = (lane >= 4 && lane < n) ? x : 0u;
    for (int offset = 16; offset; offset >>= 1) {
        const std::uint32_t other = __shfl_down_sync(kFullWarp, z, offset);
        z = add_mod(z, other, mod);
    }
    return __shfl_sync(kFullWarp, z, 0);
}

// Every lane calls this after all source values have been loaded into registers.
// Lane t returns destination value y_t in the stationary component coordinate
// order.  No component value/output/rank scratch is required.
__device__ __forceinline__ std::uint32_t apply_closed_component_warp(
    PackedWord label,
    PackedKey source,
    int n,
    int W,
    int i,
    std::uint32_t x,
    std::uint32_t mod,
    int* error
) {
    const int lane = threadIdx.x & 31;
    const std::uint32_t x0 = __shfl_sync(kFullWarp, x, 0);
    if (n == 1) return lane == 0 ? x0 : 0u;

    const std::uint32_t x1 = __shfl_sync(kFullWarp, x, 1);
    const std::uint32_t x2 = __shfl_sync(kFullWarp, x, 2);
    if (n == 3) {
        if (lane == 0) return x2;
        if (lane == 1) return add_mod(x1, x2, mod);
        if (lane == 2) return add_mod(x0, x2, mod);
        return 0u;
    }

    const std::uint32_t x3 = __shfl_sync(kFullWarp, x, 3);
    const std::uint32_t tail = warp_tail_sum_mod(x, lane, n, mod);
    const Symbol a = symbol(label, i);
    const Symbol b = symbol(label, i + 1);

    // Common destinations for all deep families.
    if (lane == 1) return add_mod(x1, x2, mod);
    if (lane == 2) return add_mod(x0, x2, mod);

    if (a == TC_L && b == TC_R) { // LR
        if (lane == 0) return x2;
        if (lane == 3) return add_mod(x3, tail, mod);
        if (lane >= 4 && lane < n) return x;
        return 0u;
    }

    if (a == TC_R && b == TC_N) { // RN
        const std::uint32_t xlast = __shfl_sync(kFullWarp, x, n - 1);
        if (lane == 0) return x3;
        if (lane == 3) return add_mod(x3, tail, mod);
        if (lane == n - 1) return add_mod(x2, xlast, mod);
        if (lane >= 4 && lane < n - 1) return x;
        return 0u;
    }

    if (a == TC_L && b == TC_N) { // LN
        PackedWord sw{};
        bool is_pivot = false;
        if (lane >= 4 && lane < n && source.type == 0) {
            sw = state_word(source, W);
            is_pivot = symbol(sw, i) == TC_L && symbol(sw, i + 1) == TC_L;
        }
        const unsigned pivot_mask = __ballot_sync(kFullWarp, is_pivot);
        if (!pivot_mask || (pivot_mask & (pivot_mask - 1))) {
            if (lane == 0) atomicCAS(error, 0, 411);
            return 0u;
        }
        const int pivot = __ffs(static_cast<int>(pivot_mask)) - 1;
        const std::uint32_t xpivot = __shfl_sync(kFullWarp, x, pivot);
        if (lane == 0) return x3;
        if (lane == 3) return add_mod(x3, tail, mod);
        if (lane == pivot) return add_mod(x2, xpivot, mod);
        if (lane >= 4 && lane < n) return x;
        return 0u;
    }

    if (lane == 0) atomicCAS(error, 0, 412);
    return 0u;
}

} // namespace oneesan::twocell::cuda_component
