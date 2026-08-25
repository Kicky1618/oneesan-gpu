#pragma once

#ifndef MASKSHARD_LOW_CLOSURE_PACKED_META
#error "packed LOW closure metadata requires MASKSHARD_LOW_CLOSURE_PACKED_META"
#endif
#ifndef MASKSHARD_LOW_CLOSURE_PACKED_PREFIX
#error "packed LOW closure metadata layers on prepacked prefixes"
#endif

// v0.42: keep v0.41's prepacked task prefix and additionally source the
// warp-uniform column begin/count and HIGH-mask base from constant memory.
// This removes per-warp reads through D_MS_LOW_CLOSURE_{BLOCK_OFF,
// COMPACT_ACTIVE_COUNT} and D_F_HIGH_MASK_OFF.
__global__ void maskshard_main_lowdesc_closure_packedmeta_inplace_kernel(
    Count* mainv, Count* blockv, Code n, int p
) {
    constexpr int H = HIGH_LUT_K;
    constexpr int FULL_CAP = (TARGET_W + 1) / 2;
    constexpr std::uint32_t HR_MASK = (1u << H) - 1u;
    const std::uint32_t pi = std::uint32_t(LOW_LUT_K - p);
    const int nb = maskshard_low_packed_main_nblocks();
    const int cap = min(D_MS_ROW_DEPTH_INDEX + 1, FULL_CAP);
    const bool saturated = cap >= FULL_CAP;

    const unsigned active = __activemask();
    const int lane = int(threadIdx.x & 31);
    const int warp_in_block = int(threadIdx.x >> 5);
    const std::uint32_t warps_per_block =
        std::uint32_t((blockDim.x + 31) >> 5);
    std::uint32_t task = std::uint32_t(blockIdx.x) * warps_per_block
                       + std::uint32_t(warp_in_block);
    const std::uint32_t task_step =
        std::uint32_t(gridDim.x) * warps_per_block;
    const std::uint32_t total =
        maskshard_low_packed_closure_prefix(int(pi), nb);

    for (; task < total; task += task_step) {
        int bid = 0;
        std::uint32_t local = 0;
        if (lane == 0) {
            int lo = 0, hi = nb + 1;
            while (lo < hi) {
                const int mid = (lo + hi) >> 1;
                if (maskshard_low_packed_closure_prefix(int(pi), mid) <= task)
                    lo = mid + 1;
                else
                    hi = mid;
            }
            bid = lo - 1;
            local = task
                - maskshard_low_packed_closure_prefix(int(pi), bid);
        }
        bid = __shfl_sync(active, bid, 0);
        local = __shfl_sync(active, local, 0);

        const FBlock x = maskshard_low_packed_main_block(bid);
        const std::uint32_t a =
            maskshard_low_packed_closure_begin(int(pi), bid);
        const std::uint32_t selected =
            maskshard_low_packed_closure_selected(int(pi), bid);
        const std::uint32_t chunks = (selected + 31u) >> 5;
        if (!chunks) continue;

        const std::uint32_t row_local = local / chunks;
        const std::uint32_t chunk = local - row_local * chunks;
        const std::uint32_t ha = maskshard_low_packed_high_mask_off(x.he);
        const std::uint32_t hr = saturated
            ? row_local
            : std::uint32_t(D_MS_LOW_CLOSURE_HIGH_COMPACT_RANK[ha + row_local]);
        const std::uint32_t qi = a + (chunk << 5) + std::uint32_t(lane);
        if (qi >= a + selected) continue;
        const std::uint32_t source = saturated
            ? D_MS_LOW_CLOSURE_COLS[qi]
            : D_MS_LOW_CLOSURE_COMPACT_COLS[qi];
        const std::uint32_t lr = lowdesc_lr(source);

        const Code i = x.off + Code(hr) * x.stride + lr;
        const Count c = mainv[i];
        if (!c) continue;

        const std::uint32_t desc = D_LOWDESC_MAIN[
            std::size_t(pi) * D_LOWDESC_MAIN_TOTAL
            + D_LOWDESC_MAIN_BASE[bid] + lr];
        const std::uint32_t kind = lowdesc_kind(desc);
        if (kind == LOWDESC_MAIN) {
            const FBlock y = maskshard_low_packed_main_block(lowdesc_block(desc));
            const Code j = y.off + Code(hr) * y.stride + lowdesc_lr(desc);
            atomic_add_mod(mainv + j, c);
        } else if (kind == LOWDESC_BLOCK) {
            const FBlock y = maskshard_low_packed_block_block(lowdesc_block(desc));
            const Code j = y.off + Code(hr) * y.stride + lowdesc_lr(desc);
            atomic_add_mod(blockv + j, c);
        } else if (kind == LOWDESC_CROSS) {
            const std::uint32_t hc = D_F_HIGH_MASK_CODES[ha + hr];
            const std::uint32_t hc2 =
                lowdesc_flip_high(hc, lowdesc_depth(desc));
            if (hc2 == 0xffffffffu) continue;
            const std::uint32_t hp = D_F_HIGH_PACKED_RANK[hc2];
            const std::uint32_t hr2 = hp & HR_MASK;
            if (p == 1) {
                const FBlock y =
                    maskshard_low_packed_main_block(lowdesc_block(desc));
                const Code j = y.off + Code(hr2) * y.stride + lowdesc_lr(desc);
                atomic_add_mod(mainv + j, c);
            } else {
                const FBlock y =
                    maskshard_low_packed_block_block(lowdesc_block(desc));
                const Code j = y.off + Code(hr2) * y.stride + lowdesc_lr(desc);
                atomic_add_mod(blockv + j, c);
            }
        }
    }
    (void)n;
}

#ifdef maskshard_main_lowdesc_closure_inplace_kernel
#undef maskshard_main_lowdesc_closure_inplace_kernel
#endif
#define maskshard_main_lowdesc_closure_inplace_kernel \
        maskshard_main_lowdesc_closure_packedmeta_inplace_kernel
