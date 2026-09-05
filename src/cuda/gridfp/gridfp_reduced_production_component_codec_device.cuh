#pragma once

#include "gridfp_reduced_production_codec_device.cuh"

#ifndef RP_FAST_COMPONENT_SUPPORT_ADJACENT_MARKS
#define RP_FAST_COMPONENT_SUPPORT_ADJACENT_MARKS 0
#endif
static_assert(RP_FAST_COMPONENT_SUPPORT_ADJACENT_MARKS == 0 ||
              RP_FAST_COMPONENT_SUPPORT_ADJACENT_MARKS == 1,
              "RP_FAST_COMPONENT_SUPPORT_ADJACENT_MARKS must be 0 or 1");

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

__device__ __forceinline__ std::uint32_t component_support_unrank_adjacent_device(
    int len,
    int ones,
    int mark0,
    int mark1,
    Rank64 rank
) {
    const int a = mark0 < mark1 ? mark0 : mark1;
    const int b = mark0 < mark1 ? mark1 : mark0;
    std::uint32_t support = 0;
    int left = ones;

    // Before the adjacent marked pair, both marks are still in the future.
    // Therefore the valid zero branch is C(rem,left)-C(rem-2,left), without
    // recomputing future_marks or testing seen_mark at every position.
    for (int pos = 0; pos < a; ++pos) {
#if RP_FAST_SUPPORT_UNRANK_EARLY_EXIT
        if (!left) break;
        const int remaining = len - pos;
        if (left == remaining) {
            support |= support_suffix_mask_device(pos, len);
            return support;
        }
        const int rem = remaining - 1;
#else
        const int rem = len - pos - 1;
#endif
        const Rank64 zero_count =
            choose_device(rem, left) - choose_device(rem - 2, left);
        if (rank < zero_count) continue;
        rank -= zero_count;
        support |= std::uint32_t(1) << pos;
        --left;
    }

#if RP_FAST_SUPPORT_UNRANK_EARLY_EXIT
    if (!left) return support;
    if (left == len - a) {
        support |= support_suffix_mask_device(a, len);
        return support;
    }
#endif

    // If the first marked bit is zero, the second one is forced to one. Its
    // whole branch has C(len-a-2,left-1) states, so we can consume the marked
    // pair in one decision and skip the forced-bit loop iteration.
    const int suffix = len - a - 2;
    const Rank64 zero_count_a = choose_device(suffix, left - 1);
    int start = b;
    if (rank < zero_count_a) {
        support |= std::uint32_t(1) << b;
        --left;
        start = b + 1;
    } else {
        rank -= zero_count_a;
        support |= std::uint32_t(1) << a;
        --left;
    }

    // Once either mark is selected, the remainder is an ordinary fixed-popcount
    // subset unrank with no marked-position constraint.
    for (int pos = start; pos < len; ++pos) {
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
        const Rank64 zero_count = choose_device(rem, left);
        if (rank < zero_count) continue;
        rank -= zero_count;
        support |= std::uint32_t(1) << pos;
        --left;
    }
    return support;
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
#if RP_FAST_COMPONENT_SUPPORT_ADJACENT_MARKS
    if ((mark0 + 1 == mark1) || (mark1 + 1 == mark0))
        return component_support_unrank_adjacent_device(
            len, ones, mark0, mark1, rank);
#endif
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
