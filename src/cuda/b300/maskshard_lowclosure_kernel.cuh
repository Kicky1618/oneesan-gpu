#pragma once

#ifdef MASKSHARD_LOW_CLOSURE_COLS

// v0.9: one warp processes up to 32 preselected closure LOW columns for one
// HIGH row. The source columns are sorted in storage all-rank order, so source
// reads remain clustered within the row instead of traversing a strided column.
__global__ void maskshard_main_lowdesc_closure_cols_inplace_kernel(
    Count* mainv, Count* blockv, Code n, int p
) {
    constexpr int S = MAXW + 2;
    constexpr uint32_t HR_MASK = (1u << HIGH_LUT_K) - 1u;
    __shared__ Code prefix[65];
    const uint32_t pi = uint32_t(LOW_LUT_K - p);
    const int nb = D_F_MAIN_NBLOCKS;

    if (threadIdx.x == 0) {
        prefix[0] = 0;
        for (int b = 0; b < nb; ++b) {
            const FBlock x = D_F_MAIN_BLOCKS[b];
            const uint32_t a = D_MS_LOW_CLOSURE_BLOCK_OFF[size_t(pi) * 65 + b];
            const uint32_t z = D_MS_LOW_CLOSURE_BLOCK_OFF[size_t(pi) * 65 + b + 1];
            const uint32_t chunks = (z - a + 31u) >> 5;
            const Code rows = x.stride ? (x.end - x.off) / x.stride : 0;
            prefix[b + 1] = prefix[b] + rows * Code(chunks);
        }
    }
    __syncthreads();

    const unsigned active = __activemask();
    const int lane = int(threadIdx.x & 31);
    const int warp_in_block = int(threadIdx.x >> 5);
    const int warps_per_block = int((blockDim.x + 31) >> 5);
    Code task = Code(blockIdx.x) * Code(warps_per_block) + Code(warp_in_block);
    const Code task_step = Code(gridDim.x) * Code(warps_per_block);
    const Code total = prefix[nb];

    for (; task < total; task += task_step) {
        int bid = 0;
        Code local = 0;
        if (lane == 0) {
            int lo = 0, hi = nb + 1;
            while (lo < hi) {
                const int mid = (lo + hi) >> 1;
                if (prefix[mid] <= task) lo = mid + 1;
                else hi = mid;
            }
            bid = lo - 1;
            local = task - prefix[bid];
        }
        bid = __shfl_sync(active, bid, 0);
        const uint32_t local_lo = __shfl_sync(active, uint32_t(local), 0);
        const uint32_t local_hi = __shfl_sync(active, uint32_t(local >> 32), 0);
        local = Code(local_lo) | (Code(local_hi) << 32);

        const uint32_t a = D_MS_LOW_CLOSURE_BLOCK_OFF[size_t(pi) * 65 + bid];
        const uint32_t z = D_MS_LOW_CLOSURE_BLOCK_OFF[size_t(pi) * 65 + bid + 1];
        const uint32_t chunks = (z - a + 31u) >> 5;
        if (!chunks) continue;
        const uint32_t hr = uint32_t(local / chunks);
        const uint32_t chunk = uint32_t(local - Code(hr) * chunks);
        const uint32_t qi = a + (chunk << 5) + uint32_t(lane);
        if (qi >= z) continue;

        const uint32_t source = D_MS_LOW_CLOSURE_COLS[qi];
        const uint32_t lr = lowdesc_lr(source);
        const FBlock x = D_F_MAIN_BLOCKS[bid];
        const Code i = x.off + Code(hr) * x.stride + lr;
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
            const uint32_t ha = D_F_HIGH_MASK_OFF[size_t(D_F_MASK) * S + x.he];
            const uint32_t hc = D_F_HIGH_MASK_CODES[ha + hr];
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
    (void)n;
}

#define maskshard_main_lowdesc_closure_inplace_kernel \
        maskshard_main_lowdesc_closure_cols_inplace_kernel

#endif

#ifdef MASKSHARD_SKIP_ZERO_BLOCK_GATHER
#include "maskshard_zero_block_gather.cuh"
#endif

#ifdef MASKSHARD_LAZY_ZERO_BLOCK_INIT
#include "maskshard_lazy_block_init.cuh"
#endif

#ifdef MASKSHARD_ROW_DEPTH_FBLOCK_IO
#include "maskshard_rowdepth_fblock_io.cuh"
#endif

#ifdef MASKSHARD_ROW_DEPTH_EXACT_IO
#include "maskshard_rowdepth_exact_io.cuh"
#endif

#ifdef MASKSHARD_ROW_DEPTH_ORBIT
#include "maskshard_rowdepth_orbit.cuh"
#endif

#ifdef MASKSHARD_ROW_DEPTH_ORBIT_COMPACT
#include "maskshard_rowdepth_orbit_compact.cuh"
#endif

#ifdef MASKSHARD_HIGH_CLOSURE_ROW_DEPTH_COMPACT
#include "maskshard_highclosure_rowdepth_compact.cuh"
#endif

#ifdef MASKSHARD_HIGH_CLOSURE_ROW_DEPTH_COMPACT_LAUNCH
#include "maskshard_highclosure_rowdepth_compact_launch.cuh"
#endif

#ifdef MASKSHARD_LOW_CLOSURE_ROW_DEPTH
#include "maskshard_lowclosure_rowdepth.cuh"
#endif

#ifdef MASKSHARD_LOW_CLOSURE_ROW_DEPTH_COMPACT
#include "maskshard_lowclosure_rowdepth_compact.cuh"
#endif

#ifdef MASKSHARD_LOW_ORBIT_ROW_DEPTH
#include "maskshard_loworbit_rowdepth.cuh"
#endif

#ifdef MASKSHARD_LOW_ORBIT_ROW_DEPTH_COMPACT
#include "maskshard_loworbit_rowdepth_compact.cuh"
#include "maskshard_loworbit_rowdepth_compact_host.cuh"
#endif

#ifdef MASKSHARD_LOW_ORBIT_ROW_DEPTH_WARP_DECODE
#include "maskshard_loworbit_rowdepth_warpdecode.cuh"
#endif

#if defined(MASKSHARD_HIGH_GROUP_SYNC) \
    && (defined(MASKSHARD_LOW_PAIR_SYNC) || defined(MASKSHARD_LOW_GROUP_SYNC))
#error "HIGH group sync interception is not combined with legacy LOW sync hooks"
#endif

#ifdef MASKSHARD_HIGH_GROUP_SYNC
#include "maskshard_high_group_sync.cuh"
#endif

#ifdef MASKSHARD_HIGH_DEAD_SYMBOL_COPIES
#include "maskshard_high_dead_symbol_copies.cuh"
#endif

#ifdef MASKSHARD_LOW_PAIR_SYNC
#include "maskshard_low_pair_sync.cuh"
#endif

#ifdef MASKSHARD_LOW_GROUP_SYNC
#include "maskshard_low_group_sync.cuh"
#endif
