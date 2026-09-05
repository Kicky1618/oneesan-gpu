#pragma once

#include "gridfp_reduced_production_owner_device.cuh"

namespace oneesan::gridfp::reducedprod {

#ifndef RP_FAST_ERASE_TWO_LOCAL_BITS
#define RP_FAST_ERASE_TWO_LOCAL_BITS 1
#endif
static_assert(RP_FAST_ERASE_TWO_LOCAL_BITS == 0 ||
              RP_FAST_ERASE_TWO_LOCAL_BITS == 1,
              "RP_FAST_ERASE_TWO_LOCAL_BITS must be 0 or 1");

struct GroupedDeviceRank {
    int owner = -1;
    Rank64 local = 0;
};

__device__ __forceinline__ Rank64 group_local_sector_offset_device(
    int L,
    int outer_ones,
    int local_ones
) {
    Rank64 off = 0;
    for (int l = 0; l < local_ones; ++l) {
        const int occupied = outer_ones + l;
        if (!(occupied & 1)) continue;
        const Rank64 pc = RP_PRIMITIVE[occupied][1];
        off += (choose_device(L, l) + choose_device(L - 2, l - 1)) * pc;
    }
    return off;
}

__device__ __forceinline__ std::uint32_t local_window_support_device(
    std::uint32_t full,
    int lo,
    int L
) {
    const std::uint32_t mask = L == 32 ? ~0u : ((std::uint32_t(1) << L) - 1u);
    return (full >> lo) & mask;
}

__device__ __forceinline__ std::uint32_t erase_two_local_bits_device(
    std::uint32_t local,
    int L,
    int a,
    int b
) {
#if RP_FAST_ERASE_TWO_LOCAL_BITS
    if (L >= 2 && L <= 32 && a >= 0 && a < L && b >= 0 && b < L && a != b) {
        const int lo = a < b ? a : b;
        const int hi = a < b ? b : a;
        const std::uint32_t width_mask = L == 32
            ? ~0u
            : ((std::uint32_t(1) << L) - 1u);
        local &= width_mask;
        const std::uint32_t low_mask = lo
            ? ((std::uint32_t(1) << lo) - 1u)
            : 0u;
        const int middle_width = hi - lo - 1;
        const std::uint32_t middle_mask = middle_width
            ? ((std::uint32_t(1) << middle_width) - 1u)
            : 0u;
        const std::uint32_t low = local & low_mask;
        const std::uint32_t middle = (local >> (lo + 1)) & middle_mask;
        const std::uint32_t high = hi == 31 ? 0u : (local >> (hi + 1));
        return low | (middle << lo) | (high << (hi - 1));
    }
#endif
    std::uint32_t compact = 0;
    int q = 0;
    for (int pos = 0; pos < L; ++pos) {
        if (pos == a || pos == b) continue;
        if ((local >> pos) & 1u) compact |= std::uint32_t(1) << q;
        ++q;
    }
    return compact;
}

// Map a reduced Q_q state into the tile-local multi-GPU storage layout.
// `owner_begin[g]` is the weighted global group prefix of GPU g's first whole
// outer-support group. The array has only ngpu 64-bit entries per tile.
__device__ __forceinline__ GroupedDeviceRank grouped_rank_device(
    DeviceKey k,
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
    const std::uint32_t full = full_support_device(k, W, q, reverse);
    const std::uint32_t outer = compact_outside_window_device(full, W, lo, hi);
    const int outer_ones = __popc(outer);
    const Rank64 group = outer_group_size_device(L, outer_ones);
    const Rank64 sr_outer = compact_support_rank_device(outer, O, outer_ones);
    Rank64 prefix = 0;
    for (int t = 0; t < outer_ones; ++t)
        prefix += choose_device(O, t) * outer_group_size_device(L, t);
    const Rank64 group_base = prefix + sr_outer * group;
    const int owner = weighted_outer_owner_device(outer, L, O, ngpu);

    const std::uint32_t local_mask = local_window_support_device(full, lo, L);
    const int local_ones = __popc(local_mask);
    const int occupied = outer_ones + local_ones;
    const Rank64 pc = RP_PRIMITIVE[occupied][1];

    // Reconstruct the full MateID only for primitive ordering. full_support_device
    // inserted N for blocked states, so the original connectivity code is still
    // available by repeating that same insertion here.
    const MateID full_mate = !k.blocked ? k.mate
        : (reverse ? blocked_exclude_reverse(k.mate, W, q)
                   : blocked_exclude(k.mate, q));
    const Rank64 pr = primitive_rank_device(full_mate, W, occupied);
    Rank64 within = group_local_sector_offset_device(L, outer_ones, local_ones);

    if (!k.blocked) {
        const Rank64 sr = compact_support_rank_device(local_mask, L, local_ones);
        within += sr * pc + pr;
    } else {
        const int missing_bit = reverse ? q - 1 : q;
        const int fixed_bit = reverse ? q : q - 1;
        const int missing_pos = missing_bit - lo;
        const int fixed_pos = fixed_bit - lo;
        const std::uint32_t compact = erase_two_local_bits_device(
            local_mask, L, missing_pos, fixed_pos);
        const Rank64 sr = compact_support_rank_device(compact, L - 2, local_ones - 1);
        within += choose_device(L, local_ones) * pc + sr * pc + pr;
    }

    return GroupedDeviceRank{owner, group_base - owner_begin[owner] + within};
}

} // namespace oneesan::gridfp::reducedprod
