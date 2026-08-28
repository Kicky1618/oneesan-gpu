#pragma once

#include "gridfp_reduced_production_owner_component_device.cuh"

namespace oneesan::gridfp::reducedprod {

struct OwnerSupportSlabDevice {
    std::uint32_t support = 0;
    std::uint8_t blocked = 0;
    std::uint8_t valid = 0;
};

__device__ __forceinline__ Rank64 owner_support_group_count_device(
    int L, int outer_ones
) {
    Rank64 total = 0;
    for (int local = 0; local <= L; ++local) {
        const int occupied = outer_ones + local;
        if (!(occupied & 1)) continue;
        total += choose_device(L, local) + choose_device(L - 2, local - 1);
    }
    return total;
}

__device__ __forceinline__ Rank64 owner_support_slab_count_device(
    int W, int K, int owner, int ngpu
) {
    const int L = K + 2;
    const int O = W - L;
    Rank64 total = 0;
    for (int r = 0; r <= O; ++r) {
        const OwnerSrRangeDevice range =
            owner_component_sr_range_device(L, O, r, owner, ngpu);
        total += (range.end - range.begin) *
                 owner_support_group_count_device(L, r);
    }
    return total;
}

__device__ __forceinline__ std::uint32_t owner_expand_blocked_local_support_device(
    std::uint32_t compact,
    int L,
    int fixed_pos,
    int missing_pos
) {
    std::uint32_t local = std::uint32_t(1) << fixed_pos;
    int cp = 0;
    for (int pos = 0; pos < L; ++pos) {
        if (pos == fixed_pos || pos == missing_pos) continue;
        if ((compact >> cp) & 1u) local |= std::uint32_t(1) << pos;
        ++cp;
    }
    return local;
}

// Unrank one support slab inside one exact outer-support group.  `outer_sr`
// is the lexicographic rank among O-bit supports with `outer_ones` bits set,
// while `within_group` enumerates the main+blocked support slabs for that
// outer support.  This is the primitive used by count-free group batches.
__device__ __forceinline__ OwnerSupportSlabDevice
owner_support_group_slab_unrank_device(
    int W,
    int q,
    bool reverse,
    int tile_start,
    int K,
    int outer_ones,
    Rank64 outer_sr,
    Rank64 within_group
) {
    const int L = K + 2;
    const int O = W - L;
    const int lo = reverse ? tile_start - 1 : tile_start - K - 1;
    const int hi = lo + L - 1;
    if (outer_ones < 0 || outer_ones > O ||
        outer_sr >= choose_device(O, outer_ones)) return {};

    const Rank64 group_slabs = owner_support_group_count_device(L, outer_ones);
    if (within_group >= group_slabs) return {};
    const std::uint32_t outer =
        support_unrank_mask_device(O, outer_ones, outer_sr);

    int local_ones = -1;
    bool blocked = false;
    Rank64 local_sr = 0;
    Rank64 within = within_group;
    for (int l = 0; l <= L; ++l) {
        const int occupied = outer_ones + l;
        if (!(occupied & 1)) continue;
        const Rank64 main_count = choose_device(L, l);
        const Rank64 block_count = choose_device(L - 2, l - 1);
        if (within < main_count) {
            local_ones = l;
            local_sr = within;
            blocked = false;
            break;
        }
        within -= main_count;
        if (within < block_count) {
            local_ones = l;
            local_sr = within;
            blocked = true;
            break;
        }
        within -= block_count;
    }
    if (local_ones < 0) return {};

    std::uint32_t local = 0;
    if (!blocked) {
        local = support_unrank_mask_device(L, local_ones, local_sr);
    } else {
        const int fixed_bit = reverse ? q : q - 1;
        const int missing_bit = reverse ? q - 1 : q;
        const int fixed_pos = fixed_bit - lo;
        const int missing_pos = missing_bit - lo;
        if (fixed_pos < 0 || fixed_pos >= L ||
            missing_pos < 0 || missing_pos >= L ||
            fixed_pos == missing_pos) return {};
        const std::uint32_t compact = support_unrank_mask_device(
            L - 2, local_ones - 1, local_sr);
        local = owner_expand_blocked_local_support_device(
            compact, L, fixed_pos, missing_pos);
    }

    std::uint32_t full = 0;
    owner_expand_outer_support_device(outer, W, lo, hi, full);
    for (int pos = 0; pos < L; ++pos)
        if ((local >> pos) & 1u)
            full |= std::uint32_t(1) << (lo + pos);

    return OwnerSupportSlabDevice{
        full, static_cast<std::uint8_t>(blocked ? 1 : 0), 1};
}

// Dense support-slab codec for one GPU's grouped-layout ownership range.
// It enumerates support masks only: every returned item denotes the contiguous
// primitive-rank slab [0, RP_PRIMITIVE[popcount(support)][1]). No MateID or
// primitive rank is materialized.
__device__ __forceinline__ OwnerSupportSlabDevice owner_support_slab_unrank_device(
    int W,
    int q,
    bool reverse,
    int tile_start,
    int K,
    int owner,
    int ngpu,
    Rank64 rank
) {
    const int L = K + 2;
    const int O = W - L;

    int outer_ones = -1;
    OwnerSrRangeDevice owner_range{};
    Rank64 slabs_per_group = 0;
    for (int r = 0; r <= O; ++r) {
        const OwnerSrRangeDevice range =
            owner_component_sr_range_device(L, O, r, owner, ngpu);
        const Rank64 group_slabs = owner_support_group_count_device(L, r);
        const Rank64 n = (range.end - range.begin) * group_slabs;
        if (rank < n) {
            outer_ones = r;
            owner_range = range;
            slabs_per_group = group_slabs;
            break;
        }
        rank -= n;
    }
    if (outer_ones < 0 || !slabs_per_group) return {};

    const Rank64 outer_sr = owner_range.begin + rank / slabs_per_group;
    const Rank64 within_group = rank % slabs_per_group;
    return owner_support_group_slab_unrank_device(
        W, q, reverse, tile_start, K,
        outer_ones, outer_sr, within_group);
}

} // namespace oneesan::gridfp::reducedprod
