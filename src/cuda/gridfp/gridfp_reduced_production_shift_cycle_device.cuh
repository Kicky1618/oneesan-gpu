#pragma once

#include "gridfp_reduced_production_equal_tile_cycle_device.cuh"

namespace oneesan::gridfp::reducedprod {

__device__ __forceinline__ std::uint32_t shift_rotate_bits_device(
    std::uint32_t x,
    int len,
    int shift
) {
    shift %= len;
    if (shift < 0) shift += len;
    if (!shift) return x;
    const std::uint32_t mask = (std::uint32_t(1) << len) - 1u;
    return ((x << shift) | (x >> (len - shift))) & mask;
}

__device__ __forceinline__ std::uint32_t shift_main_support_device(
    std::uint32_t support,
    int W,
    int Kwin,
    int S,
    bool reverse
) {
    const int span = Kwin + S + 2;
    const int lo = reverse ? 0 : W - span;
    const std::uint32_t low_mask = (std::uint32_t(1) << span) - 1u;
    const std::uint32_t span_mask = low_mask << lo;
    const std::uint32_t x = (support & span_mask) >> lo;
    const int shift = reverse ? span - S : S;
    return (support & ~span_mask) |
           (shift_rotate_bits_device(x, span, shift) << lo);
}

__device__ __forceinline__ std::uint32_t shift_blocked_support_device(
    std::uint32_t support,
    int W,
    int q,
    int Kwin,
    int S,
    bool reverse
) {
    const int span = Kwin + S + 2;
    const int compact_len = span - 2;
    const int lo = reverse ? 0 : W - span;
    std::uint32_t compact = 0;
    int cp = 0;
    for (int bit = lo; bit < lo + span; ++bit) {
        if (bit == q - 1 || bit == q) continue;
        if ((support >> bit) & 1u) compact |= std::uint32_t(1) << cp;
        ++cp;
    }
    const int shift = reverse ? compact_len - S : S;
    const std::uint32_t rotated = shift_rotate_bits_device(compact, compact_len, shift);
    std::uint32_t out = support;
    cp = 0;
    for (int bit = lo; bit < lo + span; ++bit) {
        if (bit == q - 1 || bit == q) continue;
        out &= ~(std::uint32_t(1) << bit);
        if ((rotated >> cp) & 1u) out |= std::uint32_t(1) << bit;
        ++cp;
    }
    return out;
}

__device__ __forceinline__ std::uint32_t shift_next_support_device(
    std::uint32_t support,
    bool blocked,
    int W,
    int q,
    int Kwin,
    int S,
    bool reverse
) {
    return blocked
        ? shift_blocked_support_device(support, W, q, Kwin, S, reverse)
        : shift_main_support_device(support, W, Kwin, S, reverse);
}

__device__ __forceinline__ int shift_gcd_device(int a, int b) {
    while (b) {
        const int t = a % b;
        a = b;
        b = t;
    }
    return a;
}

__device__ __forceinline__ int shift_cycle_order_device(
    bool blocked,
    int Kwin,
    int S
) {
    const int len = blocked ? Kwin + S : Kwin + S + 2;
    return len / shift_gcd_device(len, S);
}

// Returns zero for a non-leader, one for a fixed run, or the exact nontrivial
// cycle length for the lexicographically-smallest support in an orbit.
__device__ __forceinline__ int shift_cycle_leader_length_device(
    std::uint32_t support,
    bool blocked,
    int W,
    int q,
    int Kwin,
    int S,
    bool reverse
) {
    const int order = shift_cycle_order_device(blocked, Kwin, S);
    std::uint32_t cur = shift_next_support_device(
        support, blocked, W, q, Kwin, S, reverse);
    if (cur == support) return 1;
    std::uint32_t minimum = support;
    int len = 1;
    while (cur != support) {
        if (cur < minimum) minimum = cur;
        cur = shift_next_support_device(cur, blocked, W, q, Kwin, S, reverse);
        ++len;
        if (len > order) return -1;
    }
    if (minimum != support) return 0;
    return len;
}

} // namespace oneesan::gridfp::reducedprod
