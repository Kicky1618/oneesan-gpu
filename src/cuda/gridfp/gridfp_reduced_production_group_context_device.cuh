#pragma once

#include "gridfp_reduced_production_grouped_device.cuh"

namespace oneesan::gridfp::reducedprod {

#ifndef RP_RUNTIME_PRIMITIVE_RANK_SETBITS
#define RP_RUNTIME_PRIMITIVE_RANK_SETBITS 1
#endif
#ifndef RP_RUNTIME_BROADWORD_SUPPORT
#define RP_RUNTIME_BROADWORD_SUPPORT 1
#endif
#ifndef RP_RUNTIME_OWNER_FROM_BOUNDARIES
#define RP_RUNTIME_OWNER_FROM_BOUNDARIES 1
#endif
#ifndef RP_RUNTIME_SUPPORT_RANK_SETBITS
#define RP_RUNTIME_SUPPORT_RANK_SETBITS 1
#endif
#ifndef RP_RUNTIME_SECTOR_OFFSET_TABLE
#define RP_RUNTIME_SECTOR_OFFSET_TABLE 0
#endif
static_assert(RP_RUNTIME_PRIMITIVE_RANK_SETBITS == 0 || RP_RUNTIME_PRIMITIVE_RANK_SETBITS == 1,
              "RP_RUNTIME_PRIMITIVE_RANK_SETBITS must be 0 or 1");
static_assert(RP_RUNTIME_BROADWORD_SUPPORT == 0 || RP_RUNTIME_BROADWORD_SUPPORT == 1,
              "RP_RUNTIME_BROADWORD_SUPPORT must be 0 or 1");
static_assert(RP_RUNTIME_OWNER_FROM_BOUNDARIES == 0 || RP_RUNTIME_OWNER_FROM_BOUNDARIES == 1,
              "RP_RUNTIME_OWNER_FROM_BOUNDARIES must be 0 or 1");
static_assert(RP_RUNTIME_SUPPORT_RANK_SETBITS == 0 || RP_RUNTIME_SUPPORT_RANK_SETBITS == 1,
              "RP_RUNTIME_SUPPORT_RANK_SETBITS must be 0 or 1");
static_assert(RP_RUNTIME_SECTOR_OFFSET_TABLE == 0 || RP_RUNTIME_SECTOR_OFFSET_TABLE == 1,
              "RP_RUNTIME_SECTOR_OFFSET_TABLE must be 0 or 1");

static constexpr int RP_RUNTIME_SECTOR_OUTER_SLOTS = 14;
static constexpr int RP_RUNTIME_SECTOR_LOCAL_SLOTS = 16;
__device__ __constant__ std::uint32_t
RP_RUNTIME_LOCAL_SECTOR_OFFSET[RP_RUNTIME_SECTOR_OUTER_SLOTS]
                              [RP_RUNTIME_SECTOR_LOCAL_SLOTS];
static_assert(sizeof(RP_RUNTIME_LOCAL_SECTOR_OFFSET) == 896);

struct GroupedComponentContextDevice {
    int owner = -1;
    int lo = 0;
    int L = 0;
    int outer_ones = 0;
    Rank64 local_group_base = 0;
};

__device__ __forceinline__ std::uint32_t runtime_support_from_mate_device(
    MateID mate, int len
) {
#if RP_RUNTIME_BROADWORD_SUPPORT
    std::uint64_t x = (std::uint64_t(mate) | (std::uint64_t(mate) >> 1)) &
                      0x5555555555555555ULL;
    x = (x | (x >> 1)) & 0x3333333333333333ULL;
    x = (x | (x >> 2)) & 0x0f0f0f0f0f0f0f0fULL;
    x = (x | (x >> 4)) & 0x00ff00ff00ff00ffULL;
    x = (x | (x >> 8)) & 0x0000ffff0000ffffULL;
    x = (x | (x >> 16)) & 0x00000000ffffffffULL;
    std::uint32_t out = static_cast<std::uint32_t>(x);
    if (len < 32) out &= (std::uint32_t(1) << len) - 1u;
    return out;
#else
    std::uint32_t out = 0;
    for (int bit = 0; bit < len; ++bit)
        if (mget(mate, bit) != N) out |= std::uint32_t(1) << bit;
    return out;
#endif
}

__device__ __forceinline__ std::uint32_t runtime_full_support_device(
    DeviceKey k, int W, int q, bool reverse
) {
    const MateID full = !k.blocked ? k.mate
        : (reverse ? blocked_exclude_reverse(k.mate, W, q)
                   : blocked_exclude(k.mate, q));
    return runtime_support_from_mate_device(full, W);
}

__device__ __forceinline__ Rank64 runtime_compact_support_rank_device(
    std::uint32_t mask, int len, int ones
) {
#if RP_RUNTIME_SUPPORT_RANK_SETBITS
    if (len < 32) mask &= (std::uint32_t(1) << len) - 1u;
    Rank64 rank = 0;
    int left = ones;
    while (mask) {
        const int pos = __ffs(mask) - 1;
        rank += choose_device(len - pos - 1, left);
        --left;
        mask &= mask - 1u;
    }
    return rank;
#else
    return compact_support_rank_device(mask, len, ones);
#endif
}

__device__ __forceinline__ Rank64 runtime_primitive_rank_support_device(
    MateID m, int len, int occupied, std::uint32_t support
) {
#if RP_RUNTIME_PRIMITIVE_RANK_SETBITS
    int h = 1;
    int seen = 0;
    Rank64 rank = 0;
    std::uint32_t mask = support;
    if (len < 32) mask &= (std::uint32_t(1) << len) - 1u;
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

__device__ __forceinline__ Rank64 runtime_group_local_sector_offset_device(
    int L, int outer_ones, int local_ones
) {
#if RP_RUNTIME_SECTOR_OFFSET_TABLE
    return RP_RUNTIME_LOCAL_SECTOR_OFFSET[outer_ones][local_ones];
#else
    return group_local_sector_offset_device(L, outer_ones, local_ones);
#endif
}

__device__ __forceinline__ int runtime_owner_from_group_base_device(
    Rank64 group_base, int W, int K, int ngpu, const Rank64* owner_begin
) {
#if RP_RUNTIME_OWNER_FROM_BOUNDARIES
    if (W >= 8 && W <= RP_MAX_W && !(W & 1) && K == (W - 2) / 2) {
        int owner = 0;
        for (int g = 1; g < ngpu; ++g) {
            const Rank64 begin = owner_begin[g];
            if (!begin) continue;
            if (begin > group_base) break;
            owner = g;
        }
        return owner;
    }
#endif
    return -1;
}

__device__ __forceinline__ GroupedComponentContextDevice grouped_component_context_device(
    DeviceKey seed, int W, int q, bool reverse, int tile_start, int K,
    int ngpu, const Rank64* owner_begin
) {
    const int L = K + 2;
    const int O = W - L;
    const int lo = reverse ? tile_start - 1 : tile_start - K - 1;
    const int hi = lo + L - 1;
    const std::uint32_t full = runtime_full_support_device(seed, W, q, reverse);
    const std::uint32_t outer = compact_outside_window_device(full, W, lo, hi);
    const int outer_ones = __popc(outer);
    const Rank64 group = outer_group_size_device(L, outer_ones);
    const Rank64 sr_outer = runtime_compact_support_rank_device(outer, O, outer_ones);
    Rank64 prefix = 0;
    for (int t = 0; t < outer_ones; ++t)
        prefix += choose_device(O, t) * outer_group_size_device(L, t);
    const Rank64 group_base = prefix + sr_outer * group;

    int owner = runtime_owner_from_group_base_device(group_base, W, K, ngpu, owner_begin);
    if (owner < 0) owner = weighted_outer_owner_device(outer, L, O, ngpu);
    return GroupedComponentContextDevice{
        owner, lo, L, outer_ones, group_base - owner_begin[owner]};
}

__device__ __forceinline__ GroupedDeviceRank grouped_rank_in_component_device(
    DeviceKey k, int W, int q, bool reverse,
    const GroupedComponentContextDevice& ctx
) {
    const MateID full_mate = !k.blocked ? k.mate
        : (reverse ? blocked_exclude_reverse(k.mate, W, q)
                   : blocked_exclude(k.mate, q));
    const std::uint32_t full = runtime_support_from_mate_device(full_mate, W);
    const std::uint32_t local_mask = local_window_support_device(full, ctx.lo, ctx.L);
    const int local_ones = __popc(local_mask);
    const int occupied = ctx.outer_ones + local_ones;
    const Rank64 pc = RP_PRIMITIVE[occupied][1];

    const Rank64 pr = runtime_primitive_rank_support_device(full_mate, W, occupied, full);
    Rank64 within = runtime_group_local_sector_offset_device(
        ctx.L, ctx.outer_ones, local_ones);

    if (!k.blocked) {
        const Rank64 sr = runtime_compact_support_rank_device(local_mask, ctx.L, local_ones);
        within += sr * pc + pr;
    } else {
        const int missing_bit = reverse ? q - 1 : q;
        const int fixed_bit = reverse ? q : q - 1;
        const int missing_pos = missing_bit - ctx.lo;
        const int fixed_pos = fixed_bit - ctx.lo;
        const std::uint32_t compact = erase_two_local_bits_device(
            local_mask, ctx.L, missing_pos, fixed_pos);
        const Rank64 sr = runtime_compact_support_rank_device(
            compact, ctx.L - 2, local_ones - 1);
        within += choose_device(ctx.L, local_ones) * pc + sr * pc + pr;
    }
    return GroupedDeviceRank{ctx.owner, ctx.local_group_base + within};
}

} // namespace oneesan::gridfp::reducedprod
