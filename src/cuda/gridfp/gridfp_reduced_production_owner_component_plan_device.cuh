#pragma once

#include "gridfp_reduced_production_owner_component_device.cuh"
#include "gridfp_reduced_production_runtime_fastdiv64.cuh"

#ifndef RP_RUNTIME_OWNER_PREFIX_BINARY
#define RP_RUNTIME_OWNER_PREFIX_BINARY 1
#endif
#ifndef RP_RUNTIME_OWNER_PREFIX_CARRY_BEGIN
#define RP_RUNTIME_OWNER_PREFIX_CARRY_BEGIN 0
#endif
#ifndef RP_RUNTIME_OWNER_LOCAL_SECTOR_TABLE
#define RP_RUNTIME_OWNER_LOCAL_SECTOR_TABLE 1
#endif
#ifndef RP_RUNTIME_OWNER_LOCAL_SECTOR_PARITY
#define RP_RUNTIME_OWNER_LOCAL_SECTOR_PARITY 1
#endif
#ifndef RP_RUNTIME_OWNER_LOCAL_SECTOR_COMPACT
#define RP_RUNTIME_OWNER_LOCAL_SECTOR_COMPACT 0
#endif
#ifndef RP_RUNTIME_OWNER_LOCAL_SECTOR_CARRY_BEGIN
#define RP_RUNTIME_OWNER_LOCAL_SECTOR_CARRY_BEGIN 0
#endif
#ifndef RP_RUNTIME_OWNER_LOCAL_SECTOR_W28_TREE
#define RP_RUNTIME_OWNER_LOCAL_SECTOR_W28_TREE 0
#endif
static_assert(RP_RUNTIME_OWNER_PREFIX_BINARY == 0 || RP_RUNTIME_OWNER_PREFIX_BINARY == 1,
              "RP_RUNTIME_OWNER_PREFIX_BINARY must be 0 or 1");
static_assert(RP_RUNTIME_OWNER_PREFIX_CARRY_BEGIN == 0 ||
              RP_RUNTIME_OWNER_PREFIX_CARRY_BEGIN == 1,
              "RP_RUNTIME_OWNER_PREFIX_CARRY_BEGIN must be 0 or 1");
static_assert(RP_RUNTIME_OWNER_LOCAL_SECTOR_TABLE == 0 ||
              RP_RUNTIME_OWNER_LOCAL_SECTOR_TABLE == 1,
              "RP_RUNTIME_OWNER_LOCAL_SECTOR_TABLE must be 0 or 1");
static_assert(RP_RUNTIME_OWNER_LOCAL_SECTOR_PARITY == 0 ||
              RP_RUNTIME_OWNER_LOCAL_SECTOR_PARITY == 1,
              "RP_RUNTIME_OWNER_LOCAL_SECTOR_PARITY must be 0 or 1");
static_assert(RP_RUNTIME_OWNER_LOCAL_SECTOR_COMPACT == 0 ||
              RP_RUNTIME_OWNER_LOCAL_SECTOR_COMPACT == 1,
              "RP_RUNTIME_OWNER_LOCAL_SECTOR_COMPACT must be 0 or 1");
static_assert(RP_RUNTIME_OWNER_LOCAL_SECTOR_CARRY_BEGIN == 0 ||
              RP_RUNTIME_OWNER_LOCAL_SECTOR_CARRY_BEGIN == 1,
              "RP_RUNTIME_OWNER_LOCAL_SECTOR_CARRY_BEGIN must be 0 or 1");
static_assert(RP_RUNTIME_OWNER_LOCAL_SECTOR_W28_TREE == 0 ||
              RP_RUNTIME_OWNER_LOCAL_SECTOR_W28_TREE == 1,
              "RP_RUNTIME_OWNER_LOCAL_SECTOR_W28_TREE must be 0 or 1");
static_assert(!RP_RUNTIME_OWNER_LOCAL_SECTOR_COMPACT ||
              RP_RUNTIME_OWNER_LOCAL_SECTOR_PARITY,
              "compact owner local-sector table requires parity search");

namespace oneesan::gridfp::reducedprod {

#if RP_RUNTIME_OWNER_LOCAL_SECTOR_COMPACT
static constexpr int RP_RUNTIME_OWNER_LOCAL_SECTOR_POSITIVE_END_ENTRIES = 503;
__device__ __constant__ std::uint32_t
RP_RUNTIME_OWNER_LOCAL_SECTOR_POSITIVE_END[
    RP_RUNTIME_OWNER_LOCAL_SECTOR_POSITIVE_END_ENTRIES] = {
#include "gridfp_reduced_production_runtime_owner_local_sector_positive_end_values.inc"
};
static_assert(sizeof(RP_RUNTIME_OWNER_LOCAL_SECTOR_POSITIVE_END) == 2012);
#else
static constexpr int RP_RUNTIME_OWNER_LOCAL_SECTOR_END_ENTRIES = 1100;
__device__ __constant__ std::uint32_t
RP_RUNTIME_OWNER_LOCAL_SECTOR_END[RP_RUNTIME_OWNER_LOCAL_SECTOR_END_ENTRIES] = {
#include "gridfp_reduced_production_runtime_owner_local_sector_end_values.inc"
};
static_assert(sizeof(RP_RUNTIME_OWNER_LOCAL_SECTOR_END) == 4400);
#endif

// Tiny O(W) per-GPU plan.  This replaces per-component weighted-owner
// boundary division.  prefix[r] is the local component prefix before outer
// popcount r, sr_begin[r] is the first owned outer support rank in that class,
// and component_group[r] is the number of production components for one fixed
// outer support.  For W=28,K=13 this is only 3*14 uint64 values per GPU.
struct OwnerComponentPlanDevice {
    const Rank64* prefix = nullptr;          // O+2 entries
    const Rank64* sr_begin = nullptr;        // O+1 entries
    const Rank64* component_group = nullptr; // O+1 entries
};

__device__ __forceinline__ int owner_local_index_without_missing_device(
    int physical_bit, int lo, int missing_bit
) {
    return physical_bit - lo - (physical_bit > missing_bit ? 1 : 0);
}

__device__ __forceinline__ int runtime_owner_prefix_sector_device(
    const Rank64* prefix, int O, Rank64 rank
) {
#if RP_RUNTIME_OWNER_PREFIX_BINARY
    int lo = 0;
    int hi = O + 1;
    while (lo < hi) {
        const int mid = lo + ((hi - lo) >> 1);
        if (rank < prefix[mid + 1]) hi = mid;
        else lo = mid + 1;
    }
    return lo <= O ? lo : -1;
#else
    for (int r = 0; r <= O; ++r)
        if (rank < prefix[r + 1]) return r;
    return -1;
#endif
}

__device__ __forceinline__ int runtime_owner_prefix_sector_begin_device(
    const Rank64* prefix, int O, Rank64 rank, Rank64& begin
) {
#if RP_RUNTIME_OWNER_PREFIX_BINARY && RP_RUNTIME_OWNER_PREFIX_CARRY_BEGIN
    // Maintain begin == prefix[lo]. Whenever the lower bound advances, the
    // endpoint that was just loaded is exactly the new sector begin. This
    // removes the caller's post-search prefix[sector] reload, including when
    // adjacent sectors have equal prefixes because this GPU owns no supports
    // in one of the outer-popcount classes.
    int lo = 0;
    int hi = O + 1;
    begin = 0;
    while (lo < hi) {
        const int mid = lo + ((hi - lo) >> 1);
        const Rank64 end = prefix[mid + 1];
        if (rank < end) {
            hi = mid;
        } else {
            lo = mid + 1;
            begin = end;
        }
    }
    return lo <= O ? lo : -1;
#else
    const int sector = runtime_owner_prefix_sector_device(prefix, O, rank);
    if (sector < 0) {
        begin = 0;
        return -1;
    }
    begin = prefix[sector];
    return sector;
#endif
}

__device__ __forceinline__ int runtime_owner_local_sector_row_base_device(int W) {
    switch (W) {
    case 8: return 0; case 10: return 20; case 12: return 50;
    case 14: return 92; case 16: return 148; case 18: return 220;
    case 20: return 310; case 22: return 420; case 24: return 552;
    case 26: return 708; case 28: return 890; default: return -1;
    }
}

__device__ __forceinline__ int runtime_owner_local_sector_compact_width_base_device(int W) {
    switch (W) {
    case 8: return 0; case 10: return 8; case 12: return 21;
    case 14: return 39; case 16: return 64; case 18: return 96;
    case 20: return 137; case 22: return 187; case 24: return 248;
    case 26: return 320; case 28: return 405; default: return -1;
    }
}

__device__ __forceinline__ int runtime_owner_local_sector_compact_row_device(
    int W, int L, int outer_ones
) {
    const int base = runtime_owner_local_sector_compact_width_base_device(W);
    const int even_count = L >> 1;
    const int odd_count = (L - 1) >> 1;
    const int prior_even = (outer_ones + 1) >> 1;
    const int prior_odd = outer_ones >> 1;
    return base + prior_even * even_count + prior_odd * odd_count;
}

__device__ __forceinline__ Rank64 runtime_owner_local_sector_group_device(
    int W, int L, int outer_ones
) {
    const int O = W - L;
    if (L != W / 2 + 1 || outer_ones < 0 || outer_ones > O) return 0;
    const int first = (outer_ones & 1) ? 2 : 1;
    const int count = (L - first + 1) >> 1;
#if RP_RUNTIME_OWNER_LOCAL_SECTOR_COMPACT
    const int row = runtime_owner_local_sector_compact_row_device(W, L, outer_ones);
    return row >= 0 ? RP_RUNTIME_OWNER_LOCAL_SECTOR_POSITIVE_END[row + count - 1] : 0;
#else
    const int base = runtime_owner_local_sector_row_base_device(W);
    if (base < 0) return 0;
    const int row = base + outer_ones * L;
    const int last = first + ((count - 1) << 1);
    return RP_RUNTIME_OWNER_LOCAL_SECTOR_END[row + last];
#endif
}

__device__ __forceinline__ void runtime_owner_local_sector_w28_tree_device(
    int row,
    int first,
    Rank64 within,
    int& local_ones,
    Rank64& local_within
) {
    // W=28,L=15 has exactly seven positive endpoints in every outer-popcount
    // row. With the compact table those endpoints are contiguous; with the
    // legacy table they remain every other entry. Both forms use the same tree.
#if RP_RUNTIME_OWNER_LOCAL_SECTOR_COMPACT
    const Rank64 e3 = RP_RUNTIME_OWNER_LOCAL_SECTOR_POSITIVE_END[row + 3];
#else
    const Rank64 e3 = RP_RUNTIME_OWNER_LOCAL_SECTOR_END[row + first + 6];
#endif
    int index = 0;
    Rank64 begin = 0;
    if (within < e3) {
#if RP_RUNTIME_OWNER_LOCAL_SECTOR_COMPACT
        const Rank64 e1 = RP_RUNTIME_OWNER_LOCAL_SECTOR_POSITIVE_END[row + 1];
#else
        const Rank64 e1 = RP_RUNTIME_OWNER_LOCAL_SECTOR_END[row + first + 2];
#endif
        if (within < e1) {
#if RP_RUNTIME_OWNER_LOCAL_SECTOR_COMPACT
            const Rank64 e0 = RP_RUNTIME_OWNER_LOCAL_SECTOR_POSITIVE_END[row];
#else
            const Rank64 e0 = RP_RUNTIME_OWNER_LOCAL_SECTOR_END[row + first];
#endif
            if (within < e0) {
                index = 0;
            } else {
                index = 1;
                begin = e0;
            }
        } else {
#if RP_RUNTIME_OWNER_LOCAL_SECTOR_COMPACT
            const Rank64 e2 = RP_RUNTIME_OWNER_LOCAL_SECTOR_POSITIVE_END[row + 2];
#else
            const Rank64 e2 = RP_RUNTIME_OWNER_LOCAL_SECTOR_END[row + first + 4];
#endif
            if (within < e2) {
                index = 2;
                begin = e1;
            } else {
                index = 3;
                begin = e2;
            }
        }
    } else {
#if RP_RUNTIME_OWNER_LOCAL_SECTOR_COMPACT
        const Rank64 e5 = RP_RUNTIME_OWNER_LOCAL_SECTOR_POSITIVE_END[row + 5];
#else
        const Rank64 e5 = RP_RUNTIME_OWNER_LOCAL_SECTOR_END[row + first + 10];
#endif
        if (within < e5) {
#if RP_RUNTIME_OWNER_LOCAL_SECTOR_COMPACT
            const Rank64 e4 = RP_RUNTIME_OWNER_LOCAL_SECTOR_POSITIVE_END[row + 4];
#else
            const Rank64 e4 = RP_RUNTIME_OWNER_LOCAL_SECTOR_END[row + first + 8];
#endif
            if (within < e4) {
                index = 4;
                begin = e3;
            } else {
                index = 5;
                begin = e4;
            }
        } else {
            index = 6;
            begin = e5;
        }
    }
    local_ones = first + (index << 1);
    local_within = within - begin;
}

__device__ __forceinline__ bool runtime_owner_local_sector_device(
    int W,
    int L,
    int outer_ones,
    Rank64 within,
    int& local_ones,
    Rank64& local_within
) {
#if RP_RUNTIME_OWNER_LOCAL_SECTOR_TABLE
    const int O = W - L;
#if RP_RUNTIME_OWNER_LOCAL_SECTOR_COMPACT
    const int base = runtime_owner_local_sector_compact_width_base_device(W);
#else
    const int base = runtime_owner_local_sector_row_base_device(W);
#endif
    if (base >= 0 && L == W / 2 + 1 && outer_ones >= 0 && outer_ones <= O) {
#if RP_RUNTIME_OWNER_LOCAL_SECTOR_COMPACT
        const int row = runtime_owner_local_sector_compact_row_device(W, L, outer_ones);
#else
        const int row = base + outer_ones * L;
#endif
#if RP_RUNTIME_OWNER_LOCAL_SECTOR_PARITY
        // Positive local sectors require odd occupied count. local_ones=0 is
        // additionally impossible because both marked local positions must be
        // occupied, so the first positive endpoint is l=1 for even outer_ones
        // and l=2 for odd outer_ones. `within` is modulo component_group and is
        // therefore below the last positive endpoint.
        const int first = (outer_ones & 1) ? 2 : 1;
#if RP_RUNTIME_OWNER_LOCAL_SECTOR_W28_TREE
        if (W == 28 && L == 15) {
            runtime_owner_local_sector_w28_tree_device(
                row, first, within, local_ones, local_within);
            return true;
        }
#endif
        const int count = (L - first + 1) >> 1;
        int lo = 0;
        int hi = count - 1;
#if RP_RUNTIME_OWNER_LOCAL_SECTOR_CARRY_BEGIN
        Rank64 begin = 0;
        while (lo < hi) {
            const int mid = lo + ((hi - lo) >> 1);
#if RP_RUNTIME_OWNER_LOCAL_SECTOR_COMPACT
            const Rank64 end = RP_RUNTIME_OWNER_LOCAL_SECTOR_POSITIVE_END[row + mid];
#else
            const int l = first + (mid << 1);
            const Rank64 end = RP_RUNTIME_OWNER_LOCAL_SECTOR_END[row + l];
#endif
            if (within < end) {
                hi = mid;
            } else {
                lo = mid + 1;
                begin = end;
            }
        }
        local_ones = first + (lo << 1);
        local_within = within - begin;
#else
        while (lo < hi) {
            const int mid = lo + ((hi - lo) >> 1);
#if RP_RUNTIME_OWNER_LOCAL_SECTOR_COMPACT
            if (within < RP_RUNTIME_OWNER_LOCAL_SECTOR_POSITIVE_END[row + mid]) hi = mid;
#else
            const int l = first + (mid << 1);
            if (within < RP_RUNTIME_OWNER_LOCAL_SECTOR_END[row + l]) hi = mid;
#endif
            else lo = mid + 1;
        }
        const int l = first + (lo << 1);
#if RP_RUNTIME_OWNER_LOCAL_SECTOR_COMPACT
        const Rank64 begin = lo ? RP_RUNTIME_OWNER_LOCAL_SECTOR_POSITIVE_END[row + lo - 1] : 0;
#else
        const Rank64 begin = lo
            ? RP_RUNTIME_OWNER_LOCAL_SECTOR_END[row + first + ((lo - 1) << 1)]
            : 0;
#endif
        local_ones = l;
        local_within = within - begin;
#endif
#else
        int lo = 0;
        int hi = L;
        while (lo < hi) {
            const int mid = lo + ((hi - lo) >> 1);
            if (within < RP_RUNTIME_OWNER_LOCAL_SECTOR_END[row + mid]) hi = mid;
            else lo = mid + 1;
        }
        if (lo >= L) return false;
        const Rank64 begin = lo ? RP_RUNTIME_OWNER_LOCAL_SECTOR_END[row + lo - 1] : 0;
        local_ones = lo;
        local_within = within - begin;
#endif
        return true;
    }
#endif
    return false;
}

__device__ __forceinline__ MateID owner_component_label_unrank_planned_device(
    int W,
    int p,
    bool reverse,
    int tile_start,
    int K,
    const OwnerComponentPlanDevice& plan,
    Rank64 rank
) {
    const int L = K + 2;
    const int O = W - L;
    const int lo = reverse ? tile_start - 1 : tile_start - K - 1;
    const int hi = lo + L - 1;

    Rank64 prefix_begin = 0;
    const int outer_ones = runtime_owner_prefix_sector_begin_device(
        plan.prefix, O, rank, prefix_begin);
    if (outer_ones < 0) return 0;
    const Rank64 local = rank - prefix_begin;

    const Rank64 component_group = plan.component_group[outer_ones];
    if (!component_group) return 0;
    Rank64 outer_delta = 0;
    Rank64 within = 0;
#if RP_RUNTIME_FAST_DIV64
    // The fixed owner-group reciprocal table is valid exactly for the two-row
    // production geometry. Generic probes with another K retain ordinary
    // div/mod, while the primitive divisor below is universal.
    if (W >= RP_RUNTIME_OWNER_W_MIN && W <= RP_MAX_W && !(W & 1) &&
        K == (W - 2) / 2) {
        runtime_fastdivmod64_magic(
            local, component_group, runtime_owner_group_magic(W, outer_ones),
            outer_delta, within);
    } else {
        outer_delta = local / component_group;
        within = local % component_group;
    }
#else
    outer_delta = local / component_group;
    within = local % component_group;
#endif
    const Rank64 outer_sr = plan.sr_begin[outer_ones] + outer_delta;
    const std::uint32_t outer = support_unrank_mask_device(O, outer_ones, outer_sr);

    int local_ones = -1;
    Rank64 local_sr = 0;
    Rank64 primitive_rank = 0;
    Rank64 local_within = 0;
    if (runtime_owner_local_sector_device(
            W, L, outer_ones, within, local_ones, local_within)) {
        const int occupied = outer_ones + local_ones;
        const Rank64 pc = RP_PRIMITIVE[occupied][1];
        runtime_fastdivmod64_magic(
            local_within, pc, RP_RUNTIME_PRIMITIVE1_MAGIC[occupied],
            local_sr, primitive_rank);
    } else {
        Rank64 scan_within = within;
        for (int l = 0; l <= L - 1; ++l) {
            const int occupied = outer_ones + l;
            if (!(occupied & 1)) continue;
            const Rank64 pc = RP_PRIMITIVE[occupied][1];
            const Rank64 supports = choose_device(L - 1, l) - choose_device(L - 3, l);
            const Rank64 n = supports * pc;
            if (scan_within < n) {
                local_ones = l;
                runtime_fastdivmod64_magic(
                    scan_within, pc, RP_RUNTIME_PRIMITIVE1_MAGIC[occupied],
                    local_sr, primitive_rank);
                break;
            }
            scan_within -= n;
        }
    }
    if (local_ones < 0) return 0;

    const int missing = reverse ? p - 1 : p;
    const int mark_a = reverse ? p : p - 1;
    const int mark_b = reverse ? p + 1 : p - 2;
    if (missing < lo || missing > hi || mark_a < lo || mark_a > hi ||
        mark_b < lo || mark_b > hi) return 0;

    const int mark0 = owner_local_index_without_missing_device(mark_a, lo, missing);
    const int mark1 = owner_local_index_without_missing_device(mark_b, lo, missing);
    if (mark0 < 0 || mark0 >= L - 1 || mark1 < 0 || mark1 >= L - 1 || mark0 == mark1)
        return 0;

    const std::uint32_t local_support = component_support_unrank_device(
        L - 1, local_ones, mark0, mark1, local_sr);

    std::uint32_t full = 0;
    owner_expand_outer_support_device(outer, W, lo, hi, full);
    owner_expand_local_support_device(local_support, L, lo, missing, full);
    if ((full >> missing) & 1u) return 0;

    const int occupied = outer_ones + local_ones;
    const std::uint32_t label_support = owner_label_lr_support_device(full, W, missing);
    return materialize_primitive_device(label_support, W - 1, occupied, primitive_rank);
}

} // namespace oneesan::gridfp::reducedprod
