#pragma once

#include "gridfp_reduced_production_component_codec_device.cuh"

namespace oneesan::gridfp::reducedprod {

__device__ __forceinline__ Rank64 outer_group_size_device(int L, int outer_ones) {
    Rank64 total = 0;
    for (int local = 0; local <= L; ++local) {
        const int occupied = outer_ones + local;
        if (!(occupied & 1)) continue;
        const Rank64 primitive = RP_PRIMITIVE[occupied][1];
        const Rank64 main_support = choose_device(L, local);
        const Rank64 block_support = choose_device(L - 2, local - 1);
        total += (main_support + block_support) * primitive;
    }
    return total;
}

__device__ __forceinline__ Rank64 outer_group_total_device(int L, int O) {
    Rank64 total = 0;
    for (int r = 0; r <= O; ++r)
        total += choose_device(O, r) * outer_group_size_device(L, r);
    return total;
}

__device__ __forceinline__ Rank64 compact_support_rank_device(
    std::uint32_t mask, int len, int ones
) {
    Rank64 rank = 0;
    int left = ones;
    for (int pos = 0; pos < len; ++pos) {
        if (((mask >> pos) & 1u) == 0) continue;
        const int rem = len - pos - 1;
        rank += choose_device(rem, left);
        --left;
    }
    return rank;
}

__device__ __forceinline__ std::uint32_t full_support_device(DeviceKey k, int W, int q, bool reverse) {
    const MateID full = !k.blocked ? k.mate
        : (reverse ? blocked_exclude_reverse(k.mate, W, q)
                   : blocked_exclude(k.mate, q));
    std::uint32_t mask = 0;
    for (int bit = 0; bit < W; ++bit)
        if (mget(full, bit) != N) mask |= std::uint32_t(1) << bit;
    return mask;
}

__device__ __forceinline__ std::uint32_t compact_outside_window_device(
    std::uint32_t full,
    int W,
    int lo,
    int hi
) {
    std::uint32_t compact = 0;
    int q = 0;
    for (int bit = 0; bit < W; ++bit) {
        if (bit >= lo && bit <= hi) continue;
        if ((full >> bit) & 1u) compact |= std::uint32_t(1) << q;
        ++q;
    }
    return compact;
}

// Assign a complete outer-support group to one GPU.  Group midpoint ownership
// avoids splitting components and gives near-equal state counts without any
// owner table.  All arithmetic is below the W=28 reduced dimension, so 64-bit
// products remain sufficient for the supported ngpu range.
__device__ __forceinline__ int weighted_outer_owner_device(
    std::uint32_t compact_outer,
    int L,
    int O,
    int ngpu
) {
    const int r = __popc(compact_outer);
    const Rank64 group = outer_group_size_device(L, r);
    const Rank64 sr = compact_support_rank_device(compact_outer, O, r);
    Rank64 prefix = 0;
    for (int t = 0; t < r; ++t)
        prefix += choose_device(O, t) * outer_group_size_device(L, t);
    const Rank64 total = outer_group_total_device(L, O);
    const Rank64 midpoint = prefix + sr * group + group / 2;
    int owner = int((midpoint * Rank64(ngpu)) / total);
    if (owner >= ngpu) owner = ngpu - 1;
    return owner;
}

__device__ __forceinline__ int tile_owner_device(
    DeviceKey k,
    int W,
    int q,
    bool reverse,
    int tile_start,
    int K,
    int ngpu
) {
    const int L = K + 2;
    const int lo = reverse ? tile_start - 1 : tile_start - K - 1;
    const int hi = reverse ? tile_start + K : tile_start;
    const int O = W - L;
    const std::uint32_t full = full_support_device(k, W, q, reverse);
    const std::uint32_t outer = compact_outside_window_device(full, W, lo, hi);
    return weighted_outer_owner_device(outer, L, O, ngpu);
}

} // namespace oneesan::gridfp::reducedprod
