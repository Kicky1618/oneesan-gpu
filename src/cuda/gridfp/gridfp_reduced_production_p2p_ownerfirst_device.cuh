#pragma once

#include "gridfp_reduced_production_grouped_device.cuh"
#include "gridfp_reduced_production_shift_cycle_device.cuh"

namespace oneesan::gridfp::reducedprod {

static constexpr int RP_P2P_WORK_DESC_BASE_BITS = 26;
static constexpr std::uint32_t RP_P2P_WORK_DESC_BASE_MASK =
    (std::uint32_t(1) << RP_P2P_WORK_DESC_BASE_BITS) - 1u;

// Owner selection needs only the physical support.  Avoid materializing a
// MateID and avoid primitive/local rank work until the current GPU is selected
// to execute the cycle.
__device__ __forceinline__ int p2p_support_owner_device(
    std::uint32_t physical_support,
    int W,
    int tile_start,
    int K,
    bool reverse,
    int ngpu
) {
    const int L = K + 2;
    const int O = W - L;
    const int lo = reverse ? tile_start - 1 : tile_start - K - 1;
    const int hi = lo + L - 1;
    const std::uint32_t outer = compact_outside_window_device(
        physical_support, W, lo, hi);
    return weighted_outer_owner_device(outer, L, O, ngpu);
}

template<int MAX_GPU>
__device__ __forceinline__ int p2p_modal_owner_only_device(
    std::uint32_t first_support,
    bool blocked,
    int cycle_len,
    int W,
    int q,
    int Kwin,
    int S,
    bool reverse,
    int old_start,
    int ngpu
) {
    if (ngpu > MAX_GPU) return -1;
    int count[MAX_GPU]{};
    std::uint32_t cur = first_support;
    for (int h = 0; h < cycle_len; ++h) {
        const int owner = p2p_support_owner_device(
            cur, W, old_start, Kwin, reverse, ngpu);
        if (owner < 0 || owner >= ngpu) return -1;
        ++count[owner];
        cur = shift_next_support_device(
            cur, blocked, W, q, Kwin, S, reverse);
    }
    int best = 0;
    for (int g = 1; g < ngpu; ++g)
        if (count[g] > count[best]) best = g;
    return best;
}

template<int MAX_GPU, int MAX_RUNS>
__device__ __forceinline__ int p2p_modal_owner_from_support_cycle_device(
    std::uint32_t first_support,
    bool blocked,
    int cycle_len,
    int W,
    int q,
    int Kwin,
    int S,
    bool reverse,
    int old_start,
    int ngpu,
    std::uint32_t (&supports)[MAX_RUNS],
    int (&owners)[MAX_RUNS]
) {
    if (ngpu > MAX_GPU || cycle_len > MAX_RUNS) return -1;
    int count[MAX_GPU]{};
    std::uint32_t cur = first_support;
    for (int h = 0; h < cycle_len; ++h) {
        supports[h] = cur;
        const int owner = p2p_support_owner_device(
            cur, W, old_start, Kwin, reverse, ngpu);
        if (owner < 0 || owner >= ngpu) return -1;
        owners[h] = owner;
        ++count[owner];
        cur = shift_next_support_device(
            cur, blocked, W, q, Kwin, S, reverse);
    }
    int best = 0;
    for (int g = 1; g < ngpu; ++g)
        if (count[g] > count[best]) best = g;
    return best;
}

__device__ __forceinline__ std::uint32_t p2p_pack_work_descriptor_device(
    Rank64 base_rank,
    int ri
) {
    return static_cast<std::uint32_t>(base_rank) |
           (static_cast<std::uint32_t>(ri) << RP_P2P_WORK_DESC_BASE_BITS);
}

__device__ __forceinline__ Rank64 p2p_work_descriptor_base_device(
    std::uint32_t desc
) {
    return Rank64(desc & RP_P2P_WORK_DESC_BASE_MASK);
}

__device__ __forceinline__ int p2p_work_descriptor_ri_device(
    std::uint32_t desc
) {
    return int(desc >> RP_P2P_WORK_DESC_BASE_BITS);
}

} // namespace oneesan::gridfp::reducedprod
