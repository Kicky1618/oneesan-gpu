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
#ifndef RP_RUNTIME_OWNER_RECIPROCAL
#define RP_RUNTIME_OWNER_RECIPROCAL 1
#endif
#ifndef RP_RUNTIME_OWNER_FIXED54
#define RP_RUNTIME_OWNER_FIXED54 RP_RUNTIME_OWNER_RECIPROCAL
#endif
#ifndef RP_RUNTIME_SUPPORT_RANK_SETBITS
#define RP_RUNTIME_SUPPORT_RANK_SETBITS 1
#endif
#ifndef RP_RUNTIME_SECTOR_OFFSET_TABLE
#define RP_RUNTIME_SECTOR_OFFSET_TABLE 1
#endif
#ifndef RP_RUNTIME_CACHE_SECTOR_ROW_BASE
#define RP_RUNTIME_CACHE_SECTOR_ROW_BASE 1
#endif
#ifndef RP_RUNTIME_OUTER_GROUP_TABLE
#define RP_RUNTIME_OUTER_GROUP_TABLE 1
#endif
static_assert(RP_RUNTIME_PRIMITIVE_RANK_SETBITS == 0 || RP_RUNTIME_PRIMITIVE_RANK_SETBITS == 1,
              "RP_RUNTIME_PRIMITIVE_RANK_SETBITS must be 0 or 1");
static_assert(RP_RUNTIME_BROADWORD_SUPPORT == 0 || RP_RUNTIME_BROADWORD_SUPPORT == 1,
              "RP_RUNTIME_BROADWORD_SUPPORT must be 0 or 1");
static_assert(RP_RUNTIME_OWNER_FROM_BOUNDARIES == 0 || RP_RUNTIME_OWNER_FROM_BOUNDARIES == 1,
              "RP_RUNTIME_OWNER_FROM_BOUNDARIES must be 0 or 1");
static_assert(RP_RUNTIME_OWNER_RECIPROCAL == 0 || RP_RUNTIME_OWNER_RECIPROCAL == 1,
              "RP_RUNTIME_OWNER_RECIPROCAL must be 0 or 1");
static_assert(RP_RUNTIME_OWNER_FIXED54 == 0 || RP_RUNTIME_OWNER_FIXED54 == 1,
              "RP_RUNTIME_OWNER_FIXED54 must be 0 or 1");
static_assert(RP_RUNTIME_SUPPORT_RANK_SETBITS == 0 || RP_RUNTIME_SUPPORT_RANK_SETBITS == 1,
              "RP_RUNTIME_SUPPORT_RANK_SETBITS must be 0 or 1");
static_assert(RP_RUNTIME_SECTOR_OFFSET_TABLE == 0 || RP_RUNTIME_SECTOR_OFFSET_TABLE == 1,
              "RP_RUNTIME_SECTOR_OFFSET_TABLE must be 0 or 1");
static_assert(RP_RUNTIME_CACHE_SECTOR_ROW_BASE == 0 || RP_RUNTIME_CACHE_SECTOR_ROW_BASE == 1,
              "RP_RUNTIME_CACHE_SECTOR_ROW_BASE must be 0 or 1");
static_assert(RP_RUNTIME_OUTER_GROUP_TABLE == 0 || RP_RUNTIME_OUTER_GROUP_TABLE == 1,
              "RP_RUNTIME_OUTER_GROUP_TABLE must be 0 or 1");

static constexpr int RP_RUNTIME_SECTOR_TABLE_ENTRIES = 1199;
__device__ __constant__ std::uint32_t
RP_RUNTIME_LOCAL_SECTOR_OFFSET[RP_RUNTIME_SECTOR_TABLE_ENTRIES] = {
#include "gridfp_reduced_production_runtime_sector_offset_values.inc"
};
static_assert(sizeof(RP_RUNTIME_LOCAL_SECTOR_OFFSET) == 4796);

static constexpr int RP_RUNTIME_OUTER_GROUP_ENTRIES = 99;
__device__ __constant__ std::uint32_t
RP_RUNTIME_OUTER_GROUP_SIZE[RP_RUNTIME_OUTER_GROUP_ENTRIES] = {
#include "gridfp_reduced_production_runtime_outer_group_values.inc"
};
__device__ __constant__ Rank64
RP_RUNTIME_OUTER_GROUP_PREFIX[RP_RUNTIME_OUTER_GROUP_ENTRIES] = {
#include "gridfp_reduced_production_runtime_outer_prefix_values.inc"
};
static_assert(sizeof(RP_RUNTIME_OUTER_GROUP_SIZE) == 396);
static_assert(sizeof(RP_RUNTIME_OUTER_GROUP_PREFIX) == 792);

#if RP_RUNTIME_OWNER_FIXED54
__device__ __constant__ Rank64 RP_RUNTIME_OWNER_MAGIC54[11] = {
    28503795109939ULL,4047269941469ULL,555537006490ULL,
    74312840109ULL,9741361862ULL,1256304905ULL,
    159864568ULL,20116192ULL,2507347ULL,309985ULL,38053ULL
};
static_assert(sizeof(RP_RUNTIME_OWNER_MAGIC54) == 88);
#elif RP_RUNTIME_OWNER_RECIPROCAL
__device__ __constant__ Rank64 RP_RUNTIME_OWNER_TOTAL[11] = {
    632ULL,4451ULL,32427ULL,242413ULL,1849269ULL,14339193ULL,
    112685373ULL,895517316ULL,7184644894ULL,58113695597ULL,
    473397057701ULL
};
__device__ __constant__ Rank64 RP_RUNTIME_OWNER_TOTAL_MAGIC[11] = {
    29187886192578405ULL,4144404420065054ULL,568869894646732ULL,
    76096348272204ULL,9975154546856ULL,1286456223423ULL,
    163701317950ULL,20598980885ULL,2567523427ULL,317425074ULL,
    38966749ULL
};
static_assert(sizeof(RP_RUNTIME_OWNER_TOTAL) == 88);
static_assert(sizeof(RP_RUNTIME_OWNER_TOTAL_MAGIC) == 88);
#endif

static constexpr std::uint16_t RP_RUNTIME_INVALID_SECTOR_ROW = 0xffffu;
struct GroupedComponentContextDevice {
    int owner = -1;
    int lo = 0;
    int L = 0;
    std::uint16_t outer_ones = 0;
    std::uint16_t sector_row_base = RP_RUNTIME_INVALID_SECTOR_ROW;
    Rank64 local_group_base = 0;
};
static_assert(sizeof(GroupedComponentContextDevice) == 24,
              "runtime context row-base cache must not increase shared footprint");

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

__device__ __forceinline__ int runtime_sector_offset_row_base_device(int W) {
    switch (W) {
    case 8: return 0; case 10: return 24; case 12: return 59;
    case 14: return 107; case 16: return 170; case 18: return 250;
    case 20: return 349; case 22: return 469; case 24: return 612;
    case 26: return 780; case 28: return 975; default: return -1;
    }
}

__device__ __forceinline__ int runtime_outer_group_row_base_device(int W) {
    switch (W) {
    case 8: return 0; case 10: return 4; case 12: return 9;
    case 14: return 15; case 16: return 22; case 18: return 30;
    case 20: return 39; case 22: return 49; case 24: return 60;
    case 26: return 72; case 28: return 85; default: return -1;
    }
}

__device__ __forceinline__ Rank64 runtime_group_local_sector_offset_device(
    int W, std::uint16_t sector_row_base, int L, int outer_ones, int local_ones
) {
#if RP_RUNTIME_SECTOR_OFFSET_TABLE
#if RP_RUNTIME_CACHE_SECTOR_ROW_BASE
    const int base = sector_row_base == RP_RUNTIME_INVALID_SECTOR_ROW
        ? -1 : int(sector_row_base);
#else
    const int base = runtime_sector_offset_row_base_device(W);
#endif
    if (base >= 0 && L == W / 2 + 1) {
        const int index = base + outer_ones * (L + 1) + local_ones;
        return RP_RUNTIME_LOCAL_SECTOR_OFFSET[index];
    }
#endif
    return group_local_sector_offset_device(L, outer_ones, local_ones);
}

__device__ __forceinline__ bool runtime_outer_group_context_device(
    int W, int L, int O, int outer_ones, Rank64& group, Rank64& prefix
) {
#if RP_RUNTIME_OUTER_GROUP_TABLE
    const int base = runtime_outer_group_row_base_device(W);
    if (base >= 0 && L == W / 2 + 1 && O == W - L) {
        const int index = base + outer_ones;
        group = RP_RUNTIME_OUTER_GROUP_SIZE[index];
        prefix = RP_RUNTIME_OUTER_GROUP_PREFIX[index];
        return true;
    }
#endif
    group = outer_group_size_device(L, outer_ones);
    prefix = 0;
    for (int t = 0; t < outer_ones; ++t)
        prefix += choose_device(O, t) * outer_group_size_device(L, t);
    return false;
}

__device__ __forceinline__ int runtime_owner_from_group_base_device(
    Rank64 group_base, Rank64 group, int W, int K, int ngpu,
    const Rank64* owner_begin
) {
#if RP_RUNTIME_OWNER_FIXED54
    if (W >= 8 && W <= RP_MAX_W && !(W & 1) && K == (W - 2) / 2) {
        const int wi = (W - 8) >> 1;
        const Rank64 numerator =
            (group_base + group / 2) * static_cast<Rank64>(ngpu);
        Rank64 q = (numerator * RP_RUNTIME_OWNER_MAGIC54[wi]) >> 54;
        if (q >= static_cast<Rank64>(ngpu)) q = static_cast<Rank64>(ngpu - 1);
        return static_cast<int>(q);
    }
#elif RP_RUNTIME_OWNER_RECIPROCAL
    if (W >= 8 && W <= RP_MAX_W && !(W & 1) && K == (W - 2) / 2) {
        const int wi = (W - 8) >> 1;
        const Rank64 total = RP_RUNTIME_OWNER_TOTAL[wi];
        const Rank64 numerator =
            (group_base + group / 2) * static_cast<Rank64>(ngpu);
        Rank64 q = __umul64hi(numerator, RP_RUNTIME_OWNER_TOTAL_MAGIC[wi]);
        const Rank64 product_lo = q * total;
        const Rank64 product_hi = __umul64hi(q, total);
        if (product_hi || product_lo > numerator) --q;
        if (q >= static_cast<Rank64>(ngpu)) q = static_cast<Rank64>(ngpu - 1);
        return static_cast<int>(q);
    }
#endif
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
    Rank64 group = 0, prefix = 0;
    runtime_outer_group_context_device(W, L, O, outer_ones, group, prefix);
    const Rank64 sr_outer = runtime_compact_support_rank_device(outer, O, outer_ones);
    const Rank64 group_base = prefix + sr_outer * group;

    int owner = runtime_owner_from_group_base_device(
        group_base, group, W, K, ngpu, owner_begin);
    if (owner < 0) owner = weighted_outer_owner_device(outer, L, O, ngpu);
#if RP_RUNTIME_CACHE_SECTOR_ROW_BASE
    const int row = L == W / 2 + 1 ? runtime_sector_offset_row_base_device(W) : -1;
#else
    const int row = -1;
#endif
    return GroupedComponentContextDevice{
        owner,
        lo,
        L,
        static_cast<std::uint16_t>(outer_ones),
        row >= 0 ? static_cast<std::uint16_t>(row) : RP_RUNTIME_INVALID_SECTOR_ROW,
        group_base - owner_begin[owner]};
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
    const int occupied = int(ctx.outer_ones) + local_ones;
    const Rank64 pc = RP_PRIMITIVE[occupied][1];

    const Rank64 pr = runtime_primitive_rank_support_device(full_mate, W, occupied, full);
    Rank64 within = runtime_group_local_sector_offset_device(
        W, ctx.sector_row_base, ctx.L, int(ctx.outer_ones), local_ones);

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
