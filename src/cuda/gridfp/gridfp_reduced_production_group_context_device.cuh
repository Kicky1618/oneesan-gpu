#pragma once

#include "gridfp_reduced_production_grouped_device.cuh"

namespace oneesan::gridfp::reducedprod {

#ifndef RP_RUNTIME_PRIMITIVE_RANK_SETBITS
#define RP_RUNTIME_PRIMITIVE_RANK_SETBITS 1
#endif
static_assert(
    RP_RUNTIME_PRIMITIVE_RANK_SETBITS == 0 ||
    RP_RUNTIME_PRIMITIVE_RANK_SETBITS == 1,
    "RP_RUNTIME_PRIMITIVE_RANK_SETBITS must be 0 or 1");

struct GroupedComponentContextDevice {
    int owner = -1;
    int lo = 0;
    int L = 0;
    int outer_ones = 0;
    Rank64 local_group_base = 0;
};

// primitive_rank_device scans all W frontier positions and skips N. Runtime
// grouped ranking already has the exact support mask, so visit only occupied
// positions from high to low. The ordering and primitive DP updates are
// identical; only the N iterations disappear.
__device__ __forceinline__ Rank64 runtime_primitive_rank_support_device(
    MateID m,
    int len,
    int occupied,
    std::uint32_t support
) {
#if RP_RUNTIME_PRIMITIVE_RANK_SETBITS
    int h = 1;
    int seen = 0;
    Rank64 rank = 0;
    std::uint32_t mask = support;
    if (len < 32)
        mask &= (std::uint32_t(1) << len) - 1u;
    while (mask) {
        const int bit = 31 - __clz(mask);
        const MateValue c = mget(m, bit);
        const int rem = occupied - (++seen);
        if (c == L) {
            if (h > 0) rank += RP_PRIMITIVE[rem][h - 1];
            ++h;
        } else {
            --h;
        }
        mask ^= std::uint32_t(1) << bit;
    }
    return rank;
#else
    return primitive_rank_device(m, len, occupied);
#endif
}

__device__ __forceinline__ GroupedComponentContextDevice grouped_component_context_device(
    DeviceKey seed,
    int W,
    int q,
    bool reverse,
    int tile_start,
    int K,
    int ngpu,
    const Rank64* owner_begin
) {
    const int L = K + 2;
    const int O = W - L;
    const int lo = reverse ? tile_start - 1 : tile_start - K - 1;
    const int hi = lo + L - 1;
    const std::uint32_t full = full_support_device(seed, W, q, reverse);
    const std::uint32_t outer = compact_outside_window_device(full, W, lo, hi);
    const int outer_ones = __popc(outer);
    const Rank64 group = outer_group_size_device(L, outer_ones);
    const Rank64 sr_outer = compact_support_rank_device(outer, O, outer_ones);
    Rank64 prefix = 0;
    for (int t = 0; t < outer_ones; ++t)
        prefix += choose_device(O, t) * outer_group_size_device(L, t);
    const Rank64 group_base = prefix + sr_outer * group;
    const int owner = weighted_outer_owner_device(outer, L, O, ngpu);
    return GroupedComponentContextDevice{
        owner, lo, L, outer_ones, group_base - owner_begin[owner]};
}

// Fast rank inside one already reconstructed production component.  The outer
// support, its weighted owner, group prefix and outer support rank are invariant
// across the component, so they are supplied by ctx instead of recomputed for
// every source and destination lane.
__device__ __forceinline__ GroupedDeviceRank grouped_rank_in_component_device(
    DeviceKey k,
    int W,
    int q,
    bool reverse,
    const GroupedComponentContextDevice& ctx
) {
    const std::uint32_t full = full_support_device(k, W, q, reverse);
    const std::uint32_t local_mask = local_window_support_device(full, ctx.lo, ctx.L);
    const int local_ones = __popc(local_mask);
    const int occupied = ctx.outer_ones + local_ones;
    const Rank64 pc = RP_PRIMITIVE[occupied][1];

    const MateID full_mate = !k.blocked ? k.mate
        : (reverse ? blocked_exclude_reverse(k.mate, W, q)
                   : blocked_exclude(k.mate, q));
    const Rank64 pr = runtime_primitive_rank_support_device(
        full_mate, W, occupied, full);
    Rank64 within = group_local_sector_offset_device(ctx.L, ctx.outer_ones, local_ones);

    if (!k.blocked) {
        const Rank64 sr = compact_support_rank_device(local_mask, ctx.L, local_ones);
        within += sr * pc + pr;
    } else {
        const int missing_bit = reverse ? q - 1 : q;
        const int fixed_bit = reverse ? q : q - 1;
        const int missing_pos = missing_bit - ctx.lo;
        const int fixed_pos = fixed_bit - ctx.lo;
        const std::uint32_t compact = erase_two_local_bits_device(
            local_mask, ctx.L, missing_pos, fixed_pos);
        const Rank64 sr = compact_support_rank_device(compact, ctx.L - 2, local_ones - 1);
        within += choose_device(ctx.L, local_ones) * pc + sr * pc + pr;
    }
    return GroupedDeviceRank{ctx.owner, ctx.local_group_base + within};
}

} // namespace oneesan::gridfp::reducedprod
