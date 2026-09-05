#pragma once

#include "gridfp_reduced_production_shift_cycle_device.cuh"

namespace oneesan::gridfp::reducedprod {

// Enumerate every odd-cardinality full-width support exactly once.  Main
// primitive runs are indexed by support, so W=28 has exactly 2^27 such runs.
__device__ __forceinline__ std::uint32_t turn_main_support_from_rank_device(
    Rank64 rank, int W
) {
    const std::uint32_t low_mask = W <= 1 ? 0u : ((std::uint32_t(1) << (W - 1)) - 1u);
    std::uint32_t support = static_cast<std::uint32_t>(rank) & low_mask;
    if ((__popc(support) & 1) == 0)
        support |= std::uint32_t(1) << (W - 1);
    return support;
}

// B=[0,K+1] -> A=[1,K+2] at the low row edge.  Destination interval(s) in A
// equals source interval(next(s)) in B, where next is a one-bit right rotation
// of the union span [0,K+2].  This is the generic shifted-window main formula
// with reverse=true and S=1.
__device__ __forceinline__ std::uint32_t turn_main_shift_next_support_device(
    std::uint32_t support, int W, int K
) {
    return shift_main_support_device(support, W, K, 1, true);
}

__device__ __forceinline__ int turn_main_shift_cycle_order_device(int K) {
    return K + 3;
}

__device__ __forceinline__ int turn_main_shift_leader_length_device(
    std::uint32_t support, int W, int K
) {
    const int order = turn_main_shift_cycle_order_device(K);
    std::uint32_t cur = turn_main_shift_next_support_device(support, W, K);
    if (cur == support) return 1;
    std::uint32_t minimum = support;
    int len = 1;
    while (cur != support) {
        if (cur < minimum) minimum = cur;
        cur = turn_main_shift_next_support_device(cur, W, K);
        ++len;
        if (len > order) return -1;
    }
    if (minimum != support) return 0;
    return len;
}

__device__ __forceinline__ DeviceKey turn_main_run_key0_device(
    std::uint32_t support, int W
) {
    return DeviceKey{equal_materialize_full_mate0_device(support, W), 0};
}

} // namespace oneesan::gridfp::reducedprod
