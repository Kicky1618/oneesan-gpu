#pragma once

#include "gridfp_reduced_production_grouped_device.cuh"

namespace oneesan::gridfp::reducedprod {

struct EqualTileRunSeed {
    std::uint32_t support = 0;
    std::uint8_t blocked = 0;
    std::uint8_t valid = 0;
};

__device__ __forceinline__ std::uint32_t equal_expand_base_support_device(
    Rank64 compact,
    int W,
    int q
) {
    std::uint32_t full = 0;
    int cp = 0;
    for (int bit = 0; bit < W; ++bit) {
        if (bit == q - 1 || bit == q) continue;
        if ((compact >> cp) & 1ULL) full |= std::uint32_t(1) << bit;
        ++cp;
    }
    return full;
}

__device__ __forceinline__ int equal_tile_run_seeds_device(
    Rank64 compact,
    int W,
    int q,
    bool reverse,
    EqualTileRunSeed (&out)[3]
) {
    const std::uint32_t base = equal_expand_base_support_device(compact, W, q);
    const int fixed = reverse ? q : q - 1;
    const int missing = reverse ? q - 1 : q;
    const bool odd = (__popcll(compact) & 1) != 0;
    if (odd) {
        out[0] = EqualTileRunSeed{base, 0, 1};
        out[1] = EqualTileRunSeed{
            base | (std::uint32_t(1) << (q - 1)) | (std::uint32_t(1) << q), 0, 1};
        out[2] = {};
        return 2;
    }
    const std::uint32_t fixed_support = base | (std::uint32_t(1) << fixed);
    out[0] = EqualTileRunSeed{fixed_support, 0, 1};
    out[1] = EqualTileRunSeed{fixed_support, 1, 1};
    out[2] = EqualTileRunSeed{base | (std::uint32_t(1) << missing), 0, 1};
    return 3;
}

__device__ __forceinline__ std::uint32_t equal_rotate_main_support_device(
    std::uint32_t support,
    int W,
    int K,
    bool reverse
) {
    const int span = 2 * K + 2;
    const int lo = reverse ? 0 : W - span;
    const std::uint32_t low_mask = (std::uint32_t(1) << span) - 1u;
    const std::uint32_t span_mask = low_mask << lo;
    const std::uint32_t x = (support & span_mask) >> lo;
    const int shift = reverse ? span - K : K;
    const std::uint32_t y = ((x << shift) | (x >> (span - shift))) & low_mask;
    return (support & ~span_mask) | (y << lo);
}

__device__ __forceinline__ std::uint32_t equal_swap_blocked_support_device(
    std::uint32_t support,
    int W,
    int K,
    bool reverse
) {
    const int span = 2 * K + 2;
    const int lo = reverse ? 0 : W - span;
    const std::uint32_t k_mask = (std::uint32_t(1) << K) - 1u;
    const std::uint32_t low = (support >> lo) & k_mask;
    const std::uint32_t middle = (support >> (lo + K)) & 3u;
    const std::uint32_t high = (support >> (lo + K + 2)) & k_mask;
    const std::uint32_t span_mask = ((std::uint32_t(1) << span) - 1u) << lo;
    return (support & ~span_mask) |
           (high << lo) |
           (middle << (lo + K)) |
           (low << (lo + K + 2));
}

__device__ __forceinline__ std::uint32_t equal_next_support_device(
    std::uint32_t support,
    bool blocked,
    int W,
    int K,
    bool reverse
) {
    return blocked
        ? equal_swap_blocked_support_device(support, W, K, reverse)
        : equal_rotate_main_support_device(support, W, K, reverse);
}

__device__ __forceinline__ MateID equal_materialize_full_mate0_device(
    std::uint32_t physical_support,
    int W
) {
    std::uint32_t lr_support = 0;
    for (int pos = 0; pos < W; ++pos) {
        const int physical_bit = W - 1 - pos;
        if ((physical_support >> physical_bit) & 1u)
            lr_support |= std::uint32_t(1) << pos;
    }
    return materialize_primitive_device(lr_support, W, __popc(physical_support), 0);
}

__device__ __forceinline__ MateID equal_compress_blocked_device(
    MateID full,
    int W,
    int q,
    bool reverse
) {
    if (!reverse) return mshrink(full, q);
    const MateID mirrored = mirror_mate(full, W);
    const MateID compressed_mirrored = mshrink(mirrored, W - q);
    return mirror_mate(compressed_mirrored, W - 1);
}

__device__ __forceinline__ DeviceKey equal_run_key0_device(
    std::uint32_t support,
    bool blocked,
    int W,
    int q,
    bool reverse
) {
    const MateID full = equal_materialize_full_mate0_device(support, W);
    return blocked
        ? DeviceKey{equal_compress_blocked_device(full, W, q, reverse), 1}
        : DeviceKey{full, 0};
}

__device__ __forceinline__ int equal_main_cycle_order_device(int K) {
    const int a = 2 * K + 2;
    int x = a, y = K;
    while (y) {
        const int t = x % y;
        x = y;
        y = t;
    }
    return a / x;
}

// Returns zero for a non-leader. A positive return value is the exact cycle
// length. The support mask itself is a canonical, table-free cycle id.
__device__ __forceinline__ int equal_cycle_leader_length_device(
    std::uint32_t support,
    bool blocked,
    int W,
    int K,
    bool reverse
) {
    const int order = blocked ? 2 : equal_main_cycle_order_device(K);
    std::uint32_t cur = equal_next_support_device(support, blocked, W, K, reverse);
    if (cur == support) return 1;
    std::uint32_t minimum = support;
    int len = 1;
    while (cur != support) {
        if (cur < minimum) minimum = cur;
        cur = equal_next_support_device(cur, blocked, W, K, reverse);
        ++len;
        if (len > order) return -1;
    }
    if (minimum != support) return 0;
    return len;
}

} // namespace oneesan::gridfp::reducedprod
