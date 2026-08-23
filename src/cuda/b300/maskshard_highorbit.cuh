#pragma once

#include <cstdint>

// In-place HIGH-window orbit executor for fix_low=true factorized groups.
//
// A main factorized row is (HIGH all-rank, LOW mask-rank). For p in the HIGH
// window, the source pair is determined entirely by HIGH+center; LOW topology
// is untouched except by the LL boundary CROSS handled by HighDesc closure.
//
// For an orbit representative whose upper symbol is N (NN/NR/NL):
//   - companion main state is obtained by changing just the active pair;
//   - corresponding old blocked state is obtained by deleting that N;
//   - both destination HIGH ranks are dense O(1) lookups.
// This lets us update identity + include/exclude contributions in-place with
// only one M+D scratch buffer and no MateID reconstruction or Motzkin ranking.

static_assert(HIGH_LUT_K < 16, "compact HIGH+center orbit code requires HIGH<16");

__device__ __forceinline__ Count maskshard_add_mod_plain(Count a, Count b) {
    const Count mod = D_MOD;
    return a >= mod - b ? a - (mod - b) : a + b;
}

__device__ __forceinline__ uint32_t maskshard_active_high_center(
    const FBlock& x, uint32_t high_all_rank
) {
    const uint32_t hc = D_F_HIGH_ALL_CODES[D_F_HIGH_ALL_OFF[x.he] + high_all_rank];
    return (hc << 2) | uint32_t(x.c);
}

__device__ __forceinline__ MateValuePair maskshard_high_pair(
    const FBlock& x, uint32_t high_all_rank, int p
) {
    const int q = p - LOW_LUT_K; // compact active index; center is index 0
    const uint32_t active = maskshard_active_high_center(x, high_all_rank);
    return MateValuePair((active >> (2 * (q - 1))) & 15u);
}

__device__ __forceinline__ uint32_t maskshard_set_active_pair(
    uint32_t active, int p, MateValuePair v
) {
    const int q = p - LOW_LUT_K;
    const uint32_t shift = uint32_t(2 * (q - 1));
    const uint32_t z = 15u << shift;
    return (active & ~z) | (uint32_t(v) << shift);
}

__device__ __forceinline__ uint32_t maskshard_drop_active_symbol(
    uint32_t active, int p
) {
    const int q = p - LOW_LUT_K;
    const uint32_t shift = uint32_t(2 * q);
    const uint32_t lowmask = (uint32_t(1) << shift) - 1u;
    const uint32_t lo = active & lowmask;
    const uint32_t hi = active & ~lowmask;
    return lo | (hi >> 2);
}

__device__ __forceinline__ Code maskshard_main_from_active(
    uint32_t active, uint32_t low_mask_rank
) {
    constexpr int H = HIGH_LUT_K;
    const uint32_t hc = active >> 2;
    const uint32_t cv = active & 3u;
    const uint32_t packed = D_F_HIGH_PACKED_RANK[hc];
    if (packed == 0xffffffffu) return ~Code(0);
    const uint32_t hr = packed >> H;
    const int he = seg_end_height(hc, H);
    const int bid = 3 * he + int(cv);
    const FBlock y = D_F_MAIN_BLOCKS[bid];
    return y.off + Code(hr) * y.stride + low_mask_rank;
}

__device__ __forceinline__ Code maskshard_block_from_active(
    uint32_t blocked_high, uint32_t low_mask_rank
) {
    constexpr int H = HIGH_LUT_K;
    const uint32_t packed = D_F_HIGH_PACKED_RANK[blocked_high];
    if (packed == 0xffffffffu) return ~Code(0);
    const uint32_t hr = packed >> H;
    const int h = seg_end_height(blocked_high, H);
    const FBlock y = D_F_BLOCK_BLOCKS[h];
    return y.off + Code(hr) * y.stride + low_mask_rank;
}

__global__ void maskshard_main_block_highorbit_kernel(
    Count* mainv, Count* blockv, Code n, int p
) {
    Code i = Code(blockIdx.x) * blockDim.x + threadIdx.x;
    const Code step = Code(gridDim.x) * blockDim.x;
    for (; i < n; i += step) {
        const int bid = f_find_main(i);
        const FBlock x = D_F_MAIN_BLOCKS[bid];
        const Code r = i - x.off;
        const uint32_t hr = x.stride ? uint32_t(r / x.stride) : 0;
        const uint32_t lr = x.stride ? uint32_t(r - Code(hr) * x.stride) : 0;
        const uint32_t active = maskshard_active_high_center(x, hr);
        const MateValuePair w = maskshard_high_pair(x, hr, p);
        if (w != NN && w != NR && w != NL) continue;

        MateValuePair cw = LR;
        if (w == NR) cw = RN;
        else if (w == NL) cw = LN;
        const uint32_t companion_active = maskshard_set_active_pair(active, p, cw);
        const Code j = maskshard_main_from_active(companion_active, lr);
        const uint32_t dropped = maskshard_drop_active_symbol(active, p);
        const Code dj = maskshard_block_from_active(dropped, lr);
        if (j == ~Code(0) || dj == ~Code(0)) continue;

        const Count c = mainv[i];
        const Count d = blockv[dj];
        if (w == NN) {
            mainv[j] = maskshard_add_mod_plain(mainv[j], c);
            mainv[i] = maskshard_add_mod_plain(c, d);
            blockv[dj] = 0;
        } else {
            const Count cc = mainv[j];
            mainv[i] = maskshard_add_mod_plain(maskshard_add_mod_plain(c, cc), d);
            // Companion RN/LN keeps its identity value. Source NR/NL becomes
            // the new blocked contribution after old blocked is consumed.
            blockv[dj] = c;
        }
    }
}

__global__ void maskshard_main_highdesc_closure_inplace_kernel(
    Count* mainv, Count* blockv, Code n, int p
) {
    constexpr int S = MAXW + 2;
    constexpr uint32_t LR_MASK = (1u << LOW_LUT_K) - 1u;
    Code i = Code(blockIdx.x) * blockDim.x + threadIdx.x;
    const Code step = Code(gridDim.x) * blockDim.x;
    const uint32_t pi = uint32_t((TARGET_W - 1) - p);
    for (; i < n; i += step) {
        const int bid = f_find_main(i);
        const FBlock x = D_F_MAIN_BLOCKS[bid];
        const Code r = i - x.off;
        const uint32_t hr = x.stride ? uint32_t(r / x.stride) : 0;
        const uint32_t lr = x.stride ? uint32_t(r - Code(hr) * x.stride) : 0;
        const MateValuePair w = maskshard_high_pair(x, hr, p);
        if (w != LL && w != RR && w != RL) continue;

        const Count c = mainv[i];
        if (!c) continue;
        const uint32_t desc = D_HIGHDESC_MAIN[
            size_t(pi) * D_HIGHDESC_MAIN_TOTAL + D_HIGHDESC_MAIN_BASE[bid] + hr];
        const uint32_t kind = highdesc_kind(desc);
        if (kind == HIGHDESC_BLOCK) {
            const FBlock y = D_F_BLOCK_BLOCKS[highdesc_block(desc)];
            const Code j = y.off + Code(highdesc_rank(desc)) * y.stride + lr;
            atomic_add_mod(blockv + j, c);
        } else if (kind == HIGHDESC_CROSS) {
            const uint32_t a = D_F_LOW_MASK_OFF[size_t(D_F_MASK) * S + x.hs];
            const uint32_t lc = D_F_LOW_MASK_CODES[a + lr];
            const uint32_t lc2 = highdesc_flip_low(lc, highdesc_depth(desc));
            if (lc2 == 0xffffffffu) continue;
            const uint32_t lp = D_F_LOW_PACKED_RANK[lc2];
            const uint32_t lr2 = lp & LR_MASK;
            const FBlock y = D_F_BLOCK_BLOCKS[highdesc_block(desc)];
            const Code j = y.off + Code(highdesc_rank(desc)) * y.stride + lr2;
            atomic_add_mod(blockv + j, c);
        }
    }
}
