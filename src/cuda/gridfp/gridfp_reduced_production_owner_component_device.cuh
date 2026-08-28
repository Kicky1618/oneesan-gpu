#pragma once

#include "gridfp_reduced_production_grouped_device.cuh"

#ifndef RP_FAST_OWNER_SUPPORT_BITPACK
#define RP_FAST_OWNER_SUPPORT_BITPACK 1
#endif
static_assert(RP_FAST_OWNER_SUPPORT_BITPACK == 0 || RP_FAST_OWNER_SUPPORT_BITPACK == 1,
              "RP_FAST_OWNER_SUPPORT_BITPACK must be 0 or 1");

namespace oneesan::gridfp::reducedprod {

struct OwnerSrRangeDevice {
    Rank64 begin = 0;
    Rank64 end = 0;
};

__device__ __forceinline__ long long owner_ceil_div_i64(long long a, long long b) {
    if (a >= 0) return (a + b - 1) / b;
    return -((-a) / b);
}

__device__ __forceinline__ Rank64 owner_component_group_size_device(
    int L, int outer_ones
) {
    Rank64 total = 0;
    for (int local = 0; local <= L - 1; ++local) {
        const int occupied = outer_ones + local;
        if (!(occupied & 1)) continue;
        const Rank64 pc = RP_PRIMITIVE[occupied][1];
        const Rank64 supports = choose_device(L - 1, local) - choose_device(L - 3, local);
        total += supports * pc;
    }
    return total;
}

__device__ __forceinline__ OwnerSrRangeDevice owner_component_sr_range_device(
    int L, int O, int r, int owner, int ngpu
) {
    const Rank64 count = choose_device(O, r);
    const Rank64 group = outer_group_size_device(L, r);
    const Rank64 total = outer_group_total_device(L, O);
    Rank64 prefix = 0;
    for (int t = 0; t < r; ++t)
        prefix += choose_device(O, t) * outer_group_size_device(L, t);

    const long long base = static_cast<long long>(prefix + group / 2);
    const long long t0 = static_cast<long long>(
        (Rank64(owner) * total + Rank64(ngpu) - 1) / Rank64(ngpu));
    const long long t1 = static_cast<long long>(
        (Rank64(owner + 1) * total + Rank64(ngpu) - 1) / Rank64(ngpu));
    long long a = owner_ceil_div_i64(t0 - base, static_cast<long long>(group));
    long long b = owner_ceil_div_i64(t1 - base, static_cast<long long>(group));
    if (a < 0) a = 0;
    if (b < 0) b = 0;
    if (a > static_cast<long long>(count)) a = static_cast<long long>(count);
    if (b > static_cast<long long>(count)) b = static_cast<long long>(count);
    return OwnerSrRangeDevice{Rank64(a), Rank64(b)};
}

__device__ __forceinline__ Rank64 owner_component_count_device(
    int W, int K, int owner, int ngpu
) {
    const int L = K + 2;
    const int O = W - L;
    Rank64 total = 0;
    for (int r = 0; r <= O; ++r) {
        const OwnerSrRangeDevice a = owner_component_sr_range_device(L, O, r, owner, ngpu);
        total += (a.end - a.begin) * owner_component_group_size_device(L, r);
    }
    return total;
}

__device__ __forceinline__ void owner_expand_outer_support_device(
    std::uint32_t outer,
    int W,
    int lo,
    int hi,
    std::uint32_t& full
) {
#if RP_FAST_OWNER_SUPPORT_BITPACK
    const int O = W - (hi - lo + 1);
    if (O < 32) outer &= O ? ((std::uint32_t(1) << O) - 1u) : 0u;
    const std::uint32_t low_mask = lo
        ? ((std::uint32_t(1) << lo) - 1u)
        : 0u;
    full |= (outer & low_mask) | ((outer >> lo) << (hi + 1));
#else
    int q = 0;
    for (int bit = 0; bit < W; ++bit) {
        if (bit >= lo && bit <= hi) continue;
        if ((outer >> q) & 1u) full |= std::uint32_t(1) << bit;
        ++q;
    }
#endif
}

__device__ __forceinline__ void owner_expand_local_support_device(
    std::uint32_t local,
    int L,
    int lo,
    int missing_bit,
    std::uint32_t& full
) {
#if RP_FAST_OWNER_SUPPORT_BITPACK
    const int pos = missing_bit - lo;
    const int len = L - 1;
    if (len < 32) local &= len ? ((std::uint32_t(1) << len) - 1u) : 0u;
    const std::uint32_t low_mask = pos
        ? ((std::uint32_t(1) << pos) - 1u)
        : 0u;
    const std::uint32_t expanded =
        (local & low_mask) | ((local & ~low_mask) << 1);
    full |= expanded << lo;
#else
    int q = 0;
    for (int bit = lo; bit < lo + L; ++bit) {
        if (bit == missing_bit) continue;
        if ((local >> q) & 1u) full |= std::uint32_t(1) << bit;
        ++q;
    }
#endif
}

__device__ __forceinline__ std::uint32_t owner_label_lr_support_device(
    std::uint32_t full,
    int W,
    int missing_bit
) {
#if RP_FAST_OWNER_SUPPORT_BITPACK
    if (W <= 1) return 0;
    const std::uint32_t low_mask = missing_bit
        ? ((std::uint32_t(1) << missing_bit) - 1u)
        : 0u;
    const std::uint32_t compact =
        (full & low_mask) | ((full >> (missing_bit + 1)) << missing_bit);
    return __brev(compact) >> (32 - (W - 1));
#else
    std::uint32_t out = 0;
    int q = 0;
    for (int pos = 0; pos < W; ++pos) {
        const int physical_bit = W - 1 - pos;
        if (physical_bit == missing_bit) continue;
        if ((full >> physical_bit) & 1u) out |= std::uint32_t(1) << q;
        ++q;
    }
    return out;
#endif
}

// Unrank one production component label owned by `owner` for a particular
// reduced step p inside the tile.  No component/global-rank table is used.
__device__ __forceinline__ MateID owner_component_label_unrank_device(
    int W,
    int p,
    bool reverse,
    int tile_start,
    int K,
    int owner,
    int ngpu,
    Rank64 rank
) {
    const int L = K + 2;
    const int O = W - L;
    const int lo = reverse ? tile_start - 1 : tile_start - K - 1;
    const int hi = lo + L - 1;

    int outer_ones = -1;
    OwnerSrRangeDevice owner_range{};
    Rank64 component_group = 0;
    for (int r = 0; r <= O; ++r) {
        const OwnerSrRangeDevice a = owner_component_sr_range_device(L, O, r, owner, ngpu);
        const Rank64 cg = owner_component_group_size_device(L, r);
        const Rank64 n = (a.end - a.begin) * cg;
        if (rank < n) {
            outer_ones = r;
            owner_range = a;
            component_group = cg;
            break;
        }
        rank -= n;
    }
    if (outer_ones < 0 || component_group == 0) return 0;

    const Rank64 outer_sr = owner_range.begin + rank / component_group;
    Rank64 within = rank % component_group;
    const std::uint32_t outer = support_unrank_mask_device(O, outer_ones, outer_sr);

    int local_ones = -1;
    Rank64 local_sr = 0;
    Rank64 primitive_rank = 0;
    for (int l = 0; l <= L - 1; ++l) {
        const int occupied = outer_ones + l;
        if (!(occupied & 1)) continue;
        const Rank64 pc = RP_PRIMITIVE[occupied][1];
        const Rank64 supports = choose_device(L - 1, l) - choose_device(L - 3, l);
        const Rank64 n = supports * pc;
        if (within < n) {
            local_ones = l;
            local_sr = within / pc;
            primitive_rank = within % pc;
            break;
        }
        within -= n;
    }
    if (local_ones < 0) return 0;

    const int missing = reverse ? p - 1 : p;
    const int mark_a = reverse ? p : p - 1;
    const int mark_b = reverse ? p + 1 : p - 2;
    if (missing < lo || missing > hi || mark_a < lo || mark_a > hi ||
        mark_b < lo || mark_b > hi) return 0;
    const int mark0 = mark_a - lo - int(mark_a > missing);
    const int mark1 = mark_b - lo - int(mark_b > missing);
    if (mark0 < 0 || mark0 >= L - 1 || mark1 < 0 || mark1 >= L - 1 || mark0 == mark1)
        return 0;

    const std::uint32_t local = component_support_unrank_device(
        L - 1, local_ones, mark0, mark1, local_sr);
    std::uint32_t full = 0;
    owner_expand_outer_support_device(outer, W, lo, hi, full);
    owner_expand_local_support_device(local, L, lo, missing, full);
    if ((full >> missing) & 1u) return 0;

    const int occupied = outer_ones + local_ones;
    const std::uint32_t label_support = owner_label_lr_support_device(full, W, missing);
    return materialize_primitive_device(label_support, W - 1, occupied, primitive_rank);
}

} // namespace oneesan::gridfp::reducedprod
