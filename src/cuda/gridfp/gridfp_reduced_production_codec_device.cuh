#pragma once

#include "gridfp_reduced_production_device.cuh"

#ifndef RP_FAST_MATERIALIZE_PRIMITIVE_SETBITS
#define RP_FAST_MATERIALIZE_PRIMITIVE_SETBITS 1
#endif
#ifndef RP_FAST_SUPPORT_UNRANK_EARLY_EXIT
#define RP_FAST_SUPPORT_UNRANK_EARLY_EXIT 1
#endif
static_assert(RP_FAST_MATERIALIZE_PRIMITIVE_SETBITS == 0 ||
              RP_FAST_MATERIALIZE_PRIMITIVE_SETBITS == 1,
              "RP_FAST_MATERIALIZE_PRIMITIVE_SETBITS must be 0 or 1");
static_assert(RP_FAST_SUPPORT_UNRANK_EARLY_EXIT == 0 ||
              RP_FAST_SUPPORT_UNRANK_EARLY_EXIT == 1,
              "RP_FAST_SUPPORT_UNRANK_EARLY_EXIT must be 0 or 1");

namespace oneesan::gridfp::reducedprod {

__device__ __forceinline__ int sector_of_rank_device(Rank64 rank) {
    for (int p = 0; p < RP_MAX_SECTORS; ++p) {
        if (rank < RP_SECTOR_OFFSET[p + 1]) return p;
    }
    return -1;
}

__device__ __forceinline__ std::uint32_t support_suffix_mask_device(int pos, int len) {
    const int width = len - pos;
    const std::uint32_t bits = width == 32
        ? ~0u : ((std::uint32_t(1) << width) - 1u);
    return bits << pos;
}

// Support mask uses left-to-right positions in bits 0..len-1.
__device__ __forceinline__ std::uint32_t support_unrank_mask_device(int len, int ones, Rank64 rank) {
    std::uint32_t mask = 0;
    int left = ones;
    for (int pos = 0; pos < len; ++pos) {
#if RP_FAST_SUPPORT_UNRANK_EARLY_EXIT
        if (!left) break;
        const int remaining = len - pos;
        if (left == remaining) {
            mask |= support_suffix_mask_device(pos, len);
            break;
        }
        const int rem = remaining - 1;
#else
        const int rem = len - pos - 1;
#endif
        const Rank64 zero_count = RP_CHOOSE[rem][left];
        if (rank < zero_count) continue;
        rank -= zero_count;
        mask |= std::uint32_t(1) << pos;
        --left;
    }
    return mask;
}

__device__ __forceinline__ MateID materialize_primitive_device(
    std::uint32_t support, int len, int occupied, Rank64 rank
) {
    MateID m = 0;
    int h = 1;
    int seen = 0;
#if RP_FAST_MATERIALIZE_PRIMITIVE_SETBITS
    if (len < 32) support &= (std::uint32_t(1) << len) - 1u;
    while (support) {
        const int pos = __ffs(support) - 1;
        const int rem = occupied - (++seen);
        const Rank64 r_count = h > 0 ? RP_PRIMITIVE[rem][h - 1] : 0;
        MateValue v = R;
        if (rank < r_count) {
            --h;
        } else {
            rank -= r_count;
            v = L;
            ++h;
        }
        m |= MateID(v) << (2 * (len - 1 - pos));
        support &= support - 1u;
    }
#else
    for (int pos = 0; pos < len; ++pos) {
        if (((support >> pos) & 1u) == 0) continue;
        const int rem = occupied - (++seen);
        const Rank64 r_count = h > 0 ? RP_PRIMITIVE[rem][h - 1] : 0;
        MateValue v = R;
        if (rank < r_count) {
            --h;
        } else {
            rank -= r_count;
            v = L;
            ++h;
        }
        m |= MateID(v) << (2 * (len - 1 - pos));
    }
#endif
    return m;
}

__device__ __forceinline__ DeviceKey factor_unrank_device(Rank64 rank, int W, int fixed_bit) {
    const int p = sector_of_rank_device(rank);
    if (p < 0) return {};
    const int occupied = 2 * p + 1;
    const Rank64 pc = RP_SECTOR_PRIMITIVE[p];
    Rank64 local = rank - RP_SECTOR_OFFSET[p];

    if (local < RP_SECTOR_MAIN[p]) {
        const Rank64 sr = local / pc;
        const Rank64 pr = local % pc;
        const std::uint32_t support = support_unrank_mask_device(W, occupied, sr);
        return DeviceKey{materialize_primitive_device(support, W, occupied, pr), 0};
    }

    local -= RP_SECTOR_MAIN[p];
    const Rank64 sr = local / pc;
    const Rank64 pr = local % pc;
    const int len = W - 1;
    const int fixed_pos = len - 1 - fixed_bit;
    const std::uint32_t compact = support_unrank_mask_device(W - 2, occupied - 1, sr);
    std::uint32_t support = 0;
    int q = 0;
    for (int pos = 0; pos < len; ++pos) {
        if (pos == fixed_pos) {
            support |= std::uint32_t(1) << pos;
        } else {
            if ((compact >> q) & 1u) support |= std::uint32_t(1) << pos;
            ++q;
        }
    }
    return DeviceKey{materialize_primitive_device(support, len, occupied, pr), 1};
}

} // namespace oneesan::gridfp::reducedprod
