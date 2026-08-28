#pragma once

#include "gridfp_reduced_production_codec_device.cuh"

namespace oneesan::gridfp::reducedprod {

__device__ __forceinline__ Rank64 choose_device(int n, int k) {
    if (n < 0 || k < 0 || k > n) return 0;
    return RP_CHOOSE[n][k];
}

__device__ __forceinline__ Rank64 component_sector_size_device(int W, int sector) {
    const int occupied = 2 * sector + 1;
    if (occupied > W - 1) return 0;
    const Rank64 supports = choose_device(W - 1, occupied) - choose_device(W - 3, occupied);
    return supports * RP_SECTOR_PRIMITIVE[sector];
}

__device__ __forceinline__ Rank64 component_count_device(int W) {
    Rank64 total = 0;
    for (int p = 0; p < RP_MAX_SECTORS; ++p) total += component_sector_size_device(W, p);
    return total;
}

__device__ __forceinline__ int component_sector_of_rank_device(int W, Rank64& rank) {
    for (int p = 0; p < RP_MAX_SECTORS; ++p) {
        const Rank64 n = component_sector_size_device(W, p);
        if (rank < n) return p;
        rank -= n;
    }
    return -1;
}

// Lexicographic support unrank for subsets that contain at least one of the two
// marked left-to-right positions.  It has no component table and uses only the
// O(W^2) binomial table already resident for the reduced state codec.
__device__ __forceinline__ std::uint32_t component_support_unrank_device(
    int len,
    int ones,
    int mark0,
    int mark1,
    Rank64 rank
) {
    std::uint32_t support = 0;
    int left = ones;
    bool seen_mark = false;
    for (int pos = 0; pos < len; ++pos) {
#if RP_FAST_SUPPORT_UNRANK_EARLY_EXIT
        if (!left) break;
        const int remaining = len - pos;
        if (left == remaining) {
            support |= support_suffix_mask_device(pos, len);
            break;
        }
        const int rem = remaining - 1;
#else
        const int rem = len - pos - 1;
#endif
        const int future_marks = (mark0 > pos ? 1 : 0) + (mark1 > pos ? 1 : 0);
        Rank64 zero_count = choose_device(rem, left);
        if (!seen_mark) zero_count -= choose_device(rem - future_marks, left);

        if (rank < zero_count) continue;
        rank -= zero_count;
        support |= std::uint32_t(1) << pos;
        --left;
        if (pos == mark0 || pos == mark1) seen_mark = true;
    }
    return support;
}

__device__ __forceinline__ MateID component_label_unrank_device(
    int W,
    int p,
    bool reverse,
    Rank64 rank
) {
    const int len = W - 1;
    Rank64 local = rank;
    const int sector = component_sector_of_rank_device(W, local);
    if (sector < 0) return 0;
    const int occupied = 2 * sector + 1;
    const Rank64 pc = RP_SECTOR_PRIMITIVE[sector];
    const Rank64 sr = local / pc;
    const Rank64 pr = local % pc;

    const int bit0 = p - 1;
    const int bit1 = reverse ? p : p - 2;
    const int mark0 = len - 1 - bit0;
    const int mark1 = len - 1 - bit1;
    const std::uint32_t support = component_support_unrank_device(
        len, occupied, mark0, mark1, sr);
    return materialize_primitive_device(support, len, occupied, pr);
}

} // namespace oneesan::gridfp::reducedprod
