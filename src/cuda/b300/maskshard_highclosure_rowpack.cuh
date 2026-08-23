#pragma once

#ifdef MASKSHARD_HIGH_CLOSURE_ROWPACK

#ifndef MASKSHARD_HIGH_CLOSURE_ROWS
#error "MASKSHARD_HIGH_CLOSURE_ROWPACK requires MASKSHARD_HIGH_CLOSURE_ROWS"
#endif

// v0.10 research: pack selected HIGH closure rows within each source FBlock.
// The v0.8 kernel assigns one whole warp to one selected HIGH row, which wastes
// lanes whenever the current fixed LOW occupancy-mask group has fewer than 32
// columns. Here one warp owns 32 consecutive items in the compact row-major
// stream (selected-row, LOW-column). A warp may span several selected rows.
// __match_any_sync keeps the row-list and HighDesc loads at one per distinct
// row represented in that warp rather than one per lane.
__global__ void maskshard_main_highdesc_closure_rowpack_inplace_kernel(
    Count* mainv, Count* blockv, Code n, int p
) {
    constexpr int S = MAXW + 2;
    constexpr uint32_t LR_MASK = (1u << LOW_LUT_K) - 1u;
    __shared__ Code prefix[65];
    const uint32_t pi = uint32_t((TARGET_W - 1) - p);
    const int nb = D_F_MAIN_NBLOCKS;

    if (threadIdx.x == 0) {
        prefix[0] = 0;
        for (int b = 0; b < nb; ++b) {
            const FBlock x = D_F_MAIN_BLOCKS[b];
            const uint32_t a = D_HIGHDESC_CLOSURE_BLOCK_OFF[size_t(pi) * 65 + b];
            const uint32_t z = D_HIGHDESC_CLOSURE_BLOCK_OFF[size_t(pi) * 65 + b + 1];
            const Code items = x.stride ? Code(z - a) * Code(x.stride) : Code(0);
            const Code tasks = (items + 31) >> 5;
            prefix[b + 1] = prefix[b] + tasks;
        }
    }
    __syncthreads();

    const unsigned active = __activemask();
    const int lane = int(threadIdx.x & 31);
    const int warp_in_block = int(threadIdx.x >> 5);
    const int warps_per_block = int(blockDim.x >> 5);
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

        const FBlock x = D_F_MAIN_BLOCKS[bid];
        const uint32_t a = D_HIGHDESC_CLOSURE_BLOCK_OFF[size_t(pi) * 65 + bid];
        const uint32_t z = D_HIGHDESC_CLOSURE_BLOCK_OFF[size_t(pi) * 65 + bid + 1];
        const Code items = Code(z - a) * Code(x.stride);
        const Code item = (local << 5) + Code(lane);
        const bool valid = item < items;
        const unsigned valid_mask = __ballot_sync(active, valid);
        if (!valid) continue;

        const uint32_t row_local = uint32_t(item / Code(x.stride));
        const uint32_t lr = uint32_t(item - Code(row_local) * Code(x.stride));
        const unsigned row_mask = __match_any_sync(valid_mask, row_local);
        const int leader = __ffs(int(row_mask)) - 1;

        uint32_t source = 0;
        if (lane == leader)
            source = D_HIGHDESC_CLOSURE_ROWS[a + row_local];
        source = __shfl_sync(row_mask, source, leader);
        const uint32_t hr = highdesc_rank(source);

        uint32_t desc = 0;
        if (lane == leader) {
            desc = D_HIGHDESC_MAIN[
                size_t(pi) * D_HIGHDESC_MAIN_TOTAL
                + D_HIGHDESC_MAIN_BASE[bid] + hr];
        }
        desc = __shfl_sync(row_mask, desc, leader);
        const uint32_t kind = highdesc_kind(desc);

        const Code i = x.off + Code(hr) * x.stride + lr;
        const Count c = mainv[i];
        if (!c) continue;

        if (kind == HIGHDESC_BLOCK) {
            const FBlock y = D_F_BLOCK_BLOCKS[highdesc_block(desc)];
            const Code j = y.off + Code(highdesc_rank(desc)) * y.stride + lr;
            atomic_add_mod(blockv + j, c);
        } else if (kind == HIGHDESC_CROSS) {
            const uint32_t la = D_F_LOW_MASK_OFF[size_t(D_F_MASK) * S + x.hs];
            const uint32_t lc = D_F_LOW_MASK_CODES[la + lr];
            const uint32_t lc2 = highdesc_flip_low(lc, highdesc_depth(desc));
            if (lc2 == 0xffffffffu) continue;
            const uint32_t lp = D_F_LOW_PACKED_RANK[lc2];
            const uint32_t lr2 = lp & LR_MASK;
            const FBlock y = D_F_BLOCK_BLOCKS[highdesc_block(desc)];
            const Code j = y.off + Code(highdesc_rank(desc)) * y.stride + lr2;
            atomic_add_mod(blockv + j, c);
        }
    }
    (void)n;
}

#endif
