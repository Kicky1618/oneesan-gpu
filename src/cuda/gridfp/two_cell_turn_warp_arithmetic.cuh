#pragma once

#include "../../common/two_cell_turn_closed_device.cuh"
#include "two_cell_component_warp_arithmetic.cuh"

#include <cuda_runtime.h>

namespace oneesan::twocell::cuda_turn {

constexpr unsigned kFullWarp = 0xffffffffu;

__device__ __forceinline__ std::uint32_t tail_sum_from_one(
    std::uint32_t x,
    int lane,
    int n,
    std::uint32_t mod
) {
    std::uint32_t z = (lane >= 1 && lane < n) ? x : 0u;
    for (int offset = 16; offset; offset >>= 1) {
        const std::uint32_t other = __shfl_down_sync(kFullWarp, z, offset);
        if (lane + offset < 32)
            z = oneesan::twocell::cuda_component::add_mod(z, other, mod);
    }
    return __shfl_sync(kFullWarp, z, 0);
}

__device__ __forceinline__ std::uint32_t apply_closed_turn_warp(
    bool singular,
    int n,
    std::uint32_t x,
    std::uint32_t mod,
    int* error
) {
    const int lane = threadIdx.x & 31;
    const std::uint32_t x0 = __shfl_sync(kFullWarp, x, 0);
    const std::uint32_t x1 = __shfl_sync(kFullWarp, x, 1);
    const std::uint32_t x2 = __shfl_sync(kFullWarp, x, 2);

    if (singular) {
        if (n != 3) {
            if (lane == 0) atomicCAS(error, 0, 491);
            return 0;
        }
        const std::uint32_t a =
            oneesan::twocell::cuda_component::add_mod(x0, x2, mod);
        const std::uint32_t b =
            oneesan::twocell::cuda_component::add_mod(x1, x2, mod);
        if (lane == 0)
            return oneesan::twocell::cuda_component::add_mod(a, a, mod);
        if (lane == 1 || lane == 2) return b;
        return 0;
    }

    if (n < 3 || n > oneesan::twocell::kMaxTurnStates) {
        if (lane == 0) atomicCAS(error, 0, 492);
        return 0;
    }
    const std::uint32_t tail = tail_sum_from_one(x, lane, n, mod);
    if (lane == 0) {
        const std::uint32_t twice =
            oneesan::twocell::cuda_component::add_mod(x0, x0, mod);
        return oneesan::twocell::cuda_component::add_mod(twice, tail, mod);
    }
    if (lane == 1) return tail;
    if (lane >= 2 && lane < n)
        return oneesan::twocell::cuda_component::add_mod(x, x, mod);
    return 0;
}

} // namespace oneesan::twocell::cuda_turn
