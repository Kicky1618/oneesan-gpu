#pragma once

#ifndef MASKSHARD_LOW_CLOSURE_ROW_DEPTH_COMPACT
#error "maskshard_lowclosure_rowdepth_compact.cuh requires compact macro"
#endif
#ifndef MASKSHARD_LOW_CLOSURE_ROW_DEPTH
#error "exact LOW closure tasks layer on v0.24 row-depth semantics"
#endif
#ifndef MASKSHARD_LOW_CLOSURE_COLS
#error "exact LOW closure tasks require v0.9 closure columns"
#endif

__global__ void maskshard_main_lowdesc_closure_rowdepth_compact_inplace_kernel(
    Count* mainv, Count* blockv, Code n, int p
) {
    constexpr int S = MAXW + 2;
    constexpr int H = HIGH_LUT_K;
    constexpr int FULL_CAP = (TARGET_W + 1) / 2;
    constexpr int CAP_STRIDE = FULL_CAP + 1;
    constexpr std::uint32_t HR_MASK = (1u << H) - 1u;
    __shared__ Code prefix[65];
    const std::uint32_t pi = std::uint32_t(LOW_LUT_K - p);
    const int nb = D_F_MAIN_NBLOCKS;
    const int cap = min(D_MS_ROW_DEPTH_INDEX + 1, FULL_CAP);
    const bool saturated = cap >= FULL_CAP;

    if (threadIdx.x == 0) {
        prefix[0] = 0;
        for (int b = 0; b < nb; ++b) {
            const FBlock x = D_F_MAIN_BLOCKS[b];
            const std::uint32_t a = D_MS_LOW_CLOSURE_BLOCK_OFF[
                std::size_t(pi) * 65 + b];
            const std::uint32_t z = D_MS_LOW_CLOSURE_BLOCK_OFF[
                std::size_t(pi) * 65 + b + 1];
            const std::uint32_t selected = saturated
                ? z - a
                : D_MS_LOW_CLOSURE_COMPACT_ACTIVE_COUNT[
                    (std::size_t(pi) * 65 + b) * CAP_STRIDE + cap];
            const std::uint32_t rows = saturated
                ? (x.stride ? std::uint32_t((x.end - x.off) / x.stride) : 0u)
                : std::uint32_t(D_MS_LOW_CLOSURE_HIGH_ACTIVE_COUNT[
                    (std::size_t(D_F_MASK) * (H + 2) + x.he) * CAP_STRIDE + cap]);
            const std::uint32_t chunks = (selected + 31u) >> 5;
            prefix[b + 1] = prefix[b] + Code(rows) * Code(chunks);
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
        const std::uint32_t local_lo = __shfl_sync(active, std::uint32_t(local), 0);
        const std::uint32_t local_hi = __shfl_sync(active, std::uint32_t(local >> 32), 0);
        local = Code(local_lo) | (Code(local_hi) << 32);

        const FBlock x = D_F_MAIN_BLOCKS[bid];
        const std::uint32_t a = D_MS_LOW_CLOSURE_BLOCK_OFF[
            std::size_t(pi) * 65 + bid];
        const std::uint32_t z = D_MS_LOW_CLOSURE_BLOCK_OFF[
            std::size_t(pi) * 65 + bid + 1];
        const std::uint32_t selected = saturated
            ? z - a
            : D_MS_LOW_CLOSURE_COMPACT_ACTIVE_COUNT[
                (std::size_t(pi) * 65 + bid) * CAP_STRIDE + cap];
        const std::uint32_t chunks = (selected + 31u) >> 5;
        if (!chunks) continue;

        const std::uint32_t row_local = std::uint32_t(local / chunks);
        const std::uint32_t chunk = std::uint32_t(local - Code(row_local) * chunks);
        const std::uint32_t ha = D_F_HIGH_MASK_OFF[
            std::size_t(D_F_MASK) * S + x.he];
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
            std::size_t(pi) * D_LOWDESC_MAIN_TOTAL + D_LOWDESC_MAIN_BASE[bid] + lr];
        const std::uint32_t kind = lowdesc_kind(desc);
        if (kind == LOWDESC_MAIN) {
            const FBlock y = D_F_MAIN_BLOCKS[lowdesc_block(desc)];
            const Code j = y.off + Code(hr) * y.stride + lowdesc_lr(desc);
            atomic_add_mod(mainv + j, c);
        } else if (kind == LOWDESC_BLOCK) {
            const FBlock y = D_F_BLOCK_BLOCKS[lowdesc_block(desc)];
            const Code j = y.off + Code(hr) * y.stride + lowdesc_lr(desc);
            atomic_add_mod(blockv + j, c);
        } else if (kind == LOWDESC_CROSS) {
            const std::uint32_t hc = D_F_HIGH_MASK_CODES[ha + hr];
            const std::uint32_t hc2 = lowdesc_flip_high(hc, lowdesc_depth(desc));
            if (hc2 == 0xffffffffu) continue;
            const std::uint32_t hp = D_F_HIGH_PACKED_RANK[hc2];
            const std::uint32_t hr2 = hp & HR_MASK;
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

#ifdef maskshard_main_lowdesc_closure_inplace_kernel
#undef maskshard_main_lowdesc_closure_inplace_kernel
#endif
#define maskshard_main_lowdesc_closure_inplace_kernel \
        maskshard_main_lowdesc_closure_rowdepth_compact_inplace_kernel
