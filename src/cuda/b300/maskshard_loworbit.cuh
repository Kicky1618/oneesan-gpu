#pragma once

#include <cstdint>

// Zero-scratch LOW-window orbit executor for fix_low=false HIGH-mask shards.
// The authoritative HIGH-mask group itself is updated in place. HIGH exact
// topology is unchanged by orbit representatives (NN/NR/NL); boundary-crossing
// closure transitions are handled by LowDesc.

static_assert(LOW_LUT_K < 15, "compact LOW+center orbit code requires LOW<15");

__device__ __forceinline__ uint32_t maskshard_active_low_center(
    const FBlock& x, uint32_t low_all_rank
) {
    constexpr int L = LOW_LUT_K;
    const uint32_t lc = D_F_LOW_ALL_CODES[D_F_LOW_ALL_OFF[x.hs] + low_all_rank];
    return lc | (uint32_t(x.c) << (2 * L));
}

__device__ __forceinline__ MateValuePair maskshard_low_pair(
    const FBlock& x, uint32_t low_all_rank, int p
) {
    const uint32_t active = maskshard_active_low_center(x, low_all_rank);
    return MateValuePair((active >> (2 * (p - 1))) & 15u);
}

__device__ __forceinline__ uint32_t maskshard_set_low_pair(
    uint32_t active, int p, MateValuePair v
) {
    const uint32_t shift = uint32_t(2 * (p - 1));
    const uint32_t z = 15u << shift;
    return (active & ~z) | (uint32_t(v) << shift);
}

__device__ __forceinline__ uint32_t maskshard_drop_low_symbol(
    uint32_t active, int p
) {
    const uint32_t shift = uint32_t(2 * p);
    const uint32_t lowmask = (uint32_t(1) << shift) - 1u;
    return (active & lowmask) | ((active & ~lowmask) >> 2);
}

__device__ __forceinline__ Code maskshard_main_from_low_active(
    const FBlock& source, uint32_t active, uint32_t high_mask_rank
) {
    constexpr int L = LOW_LUT_K;
    constexpr uint32_t LM = (uint32_t(1) << (2 * L)) - 1u;
    const uint32_t lc = active & LM;
    const uint32_t cv = (active >> (2 * L)) & 3u;
    const uint32_t packed = D_F_LOW_PACKED_RANK[lc];
    if (packed == 0xffffffffu) return ~Code(0);
    const uint32_t lr = packed >> L;
    const int bid = 3 * int(source.he) + int(cv);
    const FBlock y = D_F_MAIN_BLOCKS[bid];
    return y.off + Code(high_mask_rank) * y.stride + lr;
}

__device__ __forceinline__ Code maskshard_block_from_low_active(
    const FBlock& source, uint32_t blocked_low, uint32_t high_mask_rank
) {
    constexpr int L = LOW_LUT_K;
    const uint32_t packed = D_F_LOW_PACKED_RANK[blocked_low];
    if (packed == 0xffffffffu) return ~Code(0);
    const uint32_t lr = packed >> L;
    const FBlock y = D_F_BLOCK_BLOCKS[source.he];
    return y.off + Code(high_mask_rank) * y.stride + lr;
}

__global__ void maskshard_main_block_loworbit_kernel(
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
        const uint32_t active = maskshard_active_low_center(x, lr);
        const MateValuePair w = maskshard_low_pair(x, lr, p);
        if (w != NN && w != NR && w != NL) continue;

        MateValuePair cw = LR;
        if (w == NR) cw = RN;
        else if (w == NL) cw = LN;
        const Code j = maskshard_main_from_low_active(
            x, maskshard_set_low_pair(active, p, cw), hr);
        const Code dj = maskshard_block_from_low_active(
            x, maskshard_drop_low_symbol(active, p), hr);
        if (j == ~Code(0) || dj == ~Code(0)) continue;

        const Count c = mainv[i];
        const Count d = blockv[dj];
        if (w == NN) {
            mainv[j] = maskshard_add_mod_plain(mainv[j], c);
            mainv[i] = maskshard_add_mod_plain(c, d);
            blockv[dj] = 0;
        } else {
            const Count cc = mainv[j];
            const Count all = maskshard_add_mod_plain(maskshard_add_mod_plain(c, cc), d);
            mainv[i] = all;
            if (p == 1) {
                mainv[j] = maskshard_add_mod_plain(c, cc);
                blockv[dj] = 0;
            } else {
                // RN/LN companion keeps identity; NR/NL becomes new blocked.
                blockv[dj] = c;
            }
        }
    }
}

__global__ void maskshard_main_lowdesc_closure_inplace_kernel(
    Count* mainv, Count* blockv, Code n, int p
) {
    constexpr int S = MAXW + 2;
    constexpr uint32_t HR_MASK = (1u << HIGH_LUT_K) - 1u;
    Code i = Code(blockIdx.x) * blockDim.x + threadIdx.x;
    const Code step = Code(gridDim.x) * blockDim.x;
    const uint32_t pi = uint32_t(LOW_LUT_K - p);
    for (; i < n; i += step) {
        const int bid = f_find_main(i);
        const FBlock x = D_F_MAIN_BLOCKS[bid];
        const Code r = i - x.off;
        const uint32_t hr = x.stride ? uint32_t(r / x.stride) : 0;
        const uint32_t lr = x.stride ? uint32_t(r - Code(hr) * x.stride) : 0;
        const MateValuePair w = maskshard_low_pair(x, lr, p);
        if (w != LL && w != RR && w != RL) continue;

        const Count c = mainv[i];
        if (!c) continue;
        const uint32_t desc = D_LOWDESC_MAIN[
            size_t(pi) * D_LOWDESC_MAIN_TOTAL + D_LOWDESC_MAIN_BASE[bid] + lr];
        const uint32_t kind = lowdesc_kind(desc);
        if (kind == LOWDESC_MAIN) {
            const FBlock y = D_F_MAIN_BLOCKS[lowdesc_block(desc)];
            const Code j = y.off + Code(hr) * y.stride + lowdesc_lr(desc);
            atomic_add_mod(mainv + j, c);
        } else if (kind == LOWDESC_BLOCK) {
            const FBlock y = D_F_BLOCK_BLOCKS[lowdesc_block(desc)];
            const Code j = y.off + Code(hr) * y.stride + lowdesc_lr(desc);
            atomic_add_mod(blockv + j, c);
        } else if (kind == LOWDESC_CROSS) {
            const uint32_t a = D_F_HIGH_MASK_OFF[size_t(D_F_MASK) * S + x.he];
            const uint32_t hc = D_F_HIGH_MASK_CODES[a + hr];
            const uint32_t hc2 = lowdesc_flip_high(hc, lowdesc_depth(desc));
            if (hc2 == 0xffffffffu) continue;
            const uint32_t hp = D_F_HIGH_PACKED_RANK[hc2];
            const uint32_t hr2 = hp & HR_MASK;
            if (p == 1) {
                const FBlock y = D_F_MAIN_BLOCKS[lowdesc_block(desc)];
                const Code j = y.off + Code(hr2) * y.stride + lowdesc_lr(desc);
                atomic_add_mod(mainv + j, c);
            } else {
                const FBlock y = D_F_BLOCK_BLOCKS[lowdesc_block(desc)];
                const Code j = y.off + Code(hr2) * y.stride + lowdesc_lr(desc);
                atomic_add_mod(blockv + j, c);
            }
        }
    }
}
