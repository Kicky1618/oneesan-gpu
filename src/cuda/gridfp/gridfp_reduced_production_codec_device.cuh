#pragma once

#include "gridfp_reduced_production_device.cuh"

namespace oneesan::gridfp::reducedprod {

__device__ __forceinline__ int sector_of_rank_device(Rank64 rank) {
    for (int p = 0; p < RP_MAX_SECTORS; ++p) {
        if (rank < RP_SECTOR_OFFSET[p + 1]) return p;
    }
    return -1;
}

// Support mask uses left-to-right positions in bits 0..len-1.
__device__ __forceinline__ std::uint32_t support_unrank_mask_device(int len, int ones, Rank64 rank) {
    std::uint32_t mask = 0;
    int left = ones;
    for (int pos = 0; pos < len; ++pos) {
        const int rem = len - pos - 1;
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
    for (int pos = 0; pos < len; ++pos) {
        if (((support >> pos) & 1u) == 0) continue;
        const int rem = occupied - (++seen);
        const Rank64 r_count = h > 0 ? RP_PRIMITIVE[rem][h - 1] : 0;
        MateValue v = R;
        if (rank < r_count) {
            v = R;
            --h;
        } else {
            rank -= r_count;
            v = L;
            ++h;
        }
        m |= MateID(v) << (2 * (len - 1 - pos));
    }
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
