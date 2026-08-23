#pragma once

#include <cstdint>
#include "maskshard_index.cuh"

#if defined(MASKSHARD_BLOCK_ORBIT) && !defined(MASKSHARD_ORBIT_AUX)
#error "MASKSHARD_BLOCK_ORBIT currently requires an orbit aux target table"
#endif

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

__device__ __forceinline__ MateValuePair maskshard_high_pair_from_active(
    uint32_t active, int p
) {
    const int q = p - LOW_LUT_K;
    return MateValuePair((active >> (2 * (q - 1))) & 15u);
}

__device__ __forceinline__ MateValuePair maskshard_high_pair(
    const FBlock& x, uint32_t high_all_rank, int p
) {
    return maskshard_high_pair_from_active(
        maskshard_active_high_center(x, high_all_rank), p);
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
    const Code step = Code(gridDim.x) * blockDim.x;
    const uint32_t pi = uint32_t((TARGET_W - 1) - p);

#ifdef MASKSHARD_BLOCK_ORBIT
    // Every blocked state has exactly one excluded target obtained by inserting
    // N at p; that target is exactly one NN/NR/NL orbit representative. Iterate
    // blocked states instead of scanning all main states and rejecting ~65%.
    const int last_bid = D_F_BLOCK_NBLOCKS - 1;
    if (last_bid < 0) return;
    const Code block_n = D_F_BLOCK_BLOCKS[last_bid].end;
    Code di = Code(blockIdx.x) * blockDim.x + threadIdx.x;
    for (; di < block_n; di += step) {
        const int dbid = f_find_block(di);
        const FBlock dx = D_F_BLOCK_BLOCKS[dbid];
        uint32_t dhr = 0, dlr = 0;
        maskshard_split_rank(di, dx, dhr, dlr);

        const size_t bdi = size_t(pi) * D_HIGHDESC_BLOCK_TOTAL
                         + D_HIGHDESC_BLOCK_BASE[dbid] + dhr;
        const uint32_t bdesc = D_HIGHDESC_BLOCK[bdi];
        if (highdesc_kind(bdesc) != HIGHDESC_MAIN) continue;
        const uint32_t sbid = highdesc_block(bdesc);
        const uint32_t shr = highdesc_rank(bdesc);
        const FBlock sx = D_F_MAIN_BLOCKS[sbid];
        const Code i = sx.off + Code(shr) * sx.stride + dlr;

        const size_t sdi = size_t(pi) * D_HIGHDESC_MAIN_TOTAL
                         + D_HIGHDESC_MAIN_BASE[sbid] + shr;
#ifdef MASKSHARD_BLOCK_ORBIT_AUX
        // v0.7 compact aux has exactly the same coordinate system as block_desc.
        const uint32_t aux = D_MS_HIGH_ORBIT_AUX[bdi];
#else
        const uint32_t aux = D_MS_HIGH_ORBIT_AUX[sdi];
#endif
        const uint32_t ak = maskshard_orbit_aux_kind(aux);
        if (ak == MS_ORBIT_AUX_INVALID) continue;

        const Count c = mainv[i];
        const Count d = blockv[di];
        if (ak == MS_ORBIT_AUX_NN) {
            const uint32_t desc = D_HIGHDESC_MAIN[sdi];
            if (highdesc_kind(desc) != HIGHDESC_MAIN) continue;
            const FBlock y = D_F_MAIN_BLOCKS[highdesc_block(desc)];
            const Code j = y.off + Code(highdesc_rank(desc)) * y.stride + dlr;
            mainv[j] = maskshard_add_mod_plain(mainv[j], c);
            mainv[i] = maskshard_add_mod_plain(c, d);
            blockv[di] = 0;
        } else {
            const FBlock y = D_F_MAIN_BLOCKS[maskshard_orbit_aux_block(aux)];
            const Code j = y.off + Code(maskshard_orbit_aux_rank(aux)) * y.stride + dlr;
            const Count cc = mainv[j];
            mainv[i] = maskshard_add_mod_plain(maskshard_add_mod_plain(c, cc), d);
            blockv[di] = c;
        }
    }
    (void)n;
#else
    Code i = Code(blockIdx.x) * blockDim.x + threadIdx.x;
    for (; i < n; i += step) {
        const int bid = f_find_main(i);
        const FBlock x = D_F_MAIN_BLOCKS[bid];
        uint32_t hr = 0, lr = 0;
        maskshard_split_rank(i, x, hr, lr);

#ifdef MASKSHARD_ORBIT_AUX
        const size_t di = size_t(pi) * D_HIGHDESC_MAIN_TOTAL
                        + D_HIGHDESC_MAIN_BASE[bid] + hr;
        const uint32_t aux = D_MS_HIGH_ORBIT_AUX[di];
        const uint32_t ak = maskshard_orbit_aux_kind(aux);
        if (ak == MS_ORBIT_AUX_INVALID) continue;
        const uint32_t desc = D_HIGHDESC_MAIN[di];

        Code j = ~Code(0), dj = ~Code(0);
        if (ak == MS_ORBIT_AUX_NN) {
            if (highdesc_kind(desc) != HIGHDESC_MAIN) continue;
            const FBlock ym = D_F_MAIN_BLOCKS[highdesc_block(desc)];
            const FBlock yd = D_F_BLOCK_BLOCKS[maskshard_orbit_aux_block(aux)];
            j = ym.off + Code(highdesc_rank(desc)) * ym.stride + lr;
            dj = yd.off + Code(maskshard_orbit_aux_rank(aux)) * yd.stride + lr;
        } else {
            if (highdesc_kind(desc) != HIGHDESC_BLOCK) continue;
            const FBlock ym = D_F_MAIN_BLOCKS[maskshard_orbit_aux_block(aux)];
            const FBlock yd = D_F_BLOCK_BLOCKS[highdesc_block(desc)];
            j = ym.off + Code(maskshard_orbit_aux_rank(aux)) * ym.stride + lr;
            dj = yd.off + Code(highdesc_rank(desc)) * yd.stride + lr;
        }

        const Count c = mainv[i];
        const Count d = blockv[dj];
        if (ak == MS_ORBIT_AUX_NN) {
            mainv[j] = maskshard_add_mod_plain(mainv[j], c);
            mainv[i] = maskshard_add_mod_plain(c, d);
            blockv[dj] = 0;
        } else {
            const Count cc = mainv[j];
            mainv[i] = maskshard_add_mod_plain(maskshard_add_mod_plain(c, cc), d);
            blockv[dj] = c;
        }
#else
        const uint32_t active = maskshard_active_high_center(x, hr);
        const MateValuePair w = maskshard_high_pair_from_active(active, p);
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
            blockv[dj] = c;
        }
#endif
    }
#endif
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
        uint32_t hr = 0, lr = 0;
        maskshard_split_rank(i, x, hr, lr);
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

#ifdef MASKSHARD_HIGH_CLOSURE_ROWS
// v0.8: HIGH transition kind is a property of (FBlock, HIGH row, p), not of
// the passive LOW column. One warp therefore classifies a HIGH row once and
// processes its LOW columns coalesced. This adds no descriptor/HBM table.
__global__ void maskshard_main_highdesc_closure_rows_inplace_kernel(
    Count* mainv, Count* blockv, Code n, int p
) {
    constexpr int S = MAXW + 2;
    constexpr uint32_t LR_MASK = (1u << LOW_LUT_K) - 1u;
    const uint32_t pi = uint32_t((TARGET_W - 1) - p);
    const unsigned active = __activemask();
    const int lane = int(threadIdx.x & 31);
    const int warp_in_block = int(threadIdx.x >> 5);
    const int warps_per_block = int((blockDim.x + 31) >> 5);
    Code row = Code(blockIdx.x) * Code(warps_per_block) + Code(warp_in_block);
    const Code row_step = Code(gridDim.x) * Code(warps_per_block);

    for (;; row += row_step) {
        int bid = -1;
        uint32_t hr = 0;
        if (lane == 0) {
            Code r = row;
            for (int b = 0; b < D_F_MAIN_NBLOCKS; ++b) {
                const FBlock z = D_F_MAIN_BLOCKS[b];
                if (!z.stride) continue;
                const Code rows = (z.end - z.off) / Code(z.stride);
                if (r < rows) {
                    bid = b;
                    hr = uint32_t(r);
                    break;
                }
                r -= rows;
            }
        }
        bid = __shfl_sync(active, bid, 0);
        if (bid < 0) break;
        hr = __shfl_sync(active, hr, 0);
        const FBlock x = D_F_MAIN_BLOCKS[bid];

        int is_closure = 0;
        uint32_t desc = 0;
        if (lane == 0) {
            const MateValuePair w = maskshard_high_pair(x, hr, p);
            is_closure = (w == LL || w == RR || w == RL);
            if (is_closure) {
                desc = D_HIGHDESC_MAIN[
                    size_t(pi) * D_HIGHDESC_MAIN_TOTAL
                    + D_HIGHDESC_MAIN_BASE[bid] + hr];
            }
        }
        is_closure = __shfl_sync(active, is_closure, 0);
        if (!is_closure) continue;
        desc = __shfl_sync(active, desc, 0);
        const uint32_t kind = highdesc_kind(desc);
        if (kind != HIGHDESC_BLOCK && kind != HIGHDESC_CROSS) continue;

        for (uint32_t lr = uint32_t(lane); lr < x.stride; lr += 32u) {
            const Code i = x.off + Code(hr) * x.stride + lr;
            const Count c = mainv[i];
            if (!c) continue;
            if (kind == HIGHDESC_BLOCK) {
                const FBlock y = D_F_BLOCK_BLOCKS[highdesc_block(desc)];
                const Code j = y.off + Code(highdesc_rank(desc)) * y.stride + lr;
                atomic_add_mod(blockv + j, c);
            } else {
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
    (void)n;
}

// The batch source keeps calling the historical symbol; select the row executor
// at preprocessing time without touching v0.4-v0.7 call sites.
#define maskshard_main_highdesc_closure_inplace_kernel \
        maskshard_main_highdesc_closure_rows_inplace_kernel
#endif
