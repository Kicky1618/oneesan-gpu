#pragma once

#ifndef MASKSHARD_LOW_ORBIT_ROW_DEPTH_WARP_DECODE
#error "warp LOW orbit decode header requires MASKSHARD_LOW_ORBIT_ROW_DEPTH_WARP_DECODE"
#endif
#ifndef MASKSHARD_LOW_ORBIT_ROW_DEPTH_COMPACT
#error "warp LOW orbit decode layers on exact compact LOW orbit tasks"
#endif

__device__ __forceinline__ int maskshard_loworbit_compact_find_block(
    Code task, int nb
) {
    int lo = 0, hi = nb + 1;
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (D_MS_LOW_ORBIT_COMPACT_TASK_PREFIX[mid] <= task) lo = mid + 1;
        else hi = mid;
    }
    return lo - 1;
}

__global__ void maskshard_main_block_loworbit_rowdepth_warpdecode_kernel(
    Count* mainv, Count* blockv, Code n, int p
) {
    constexpr int S = MAXW + 2;
    constexpr int FULL_CAP = TARGET_W / 2;
    const int nb = D_F_BLOCK_NBLOCKS;
    if (nb <= 0) return;
    const int cap = min(D_MS_ROW_DEPTH_INDEX + 1, FULL_CAP);
    const bool saturated = cap >= FULL_CAP;
    const Code total = saturated
        ? D_F_BLOCK_BLOCKS[nb - 1].end
        : D_MS_LOW_ORBIT_COMPACT_TASK_PREFIX[nb];
    const std::uint32_t pi = std::uint32_t(LOW_LUT_K - p);
    Code task = Code(blockIdx.x) * blockDim.x + threadIdx.x;
    const Code step = Code(gridDim.x) * blockDim.x;

    for (; task < total; task += step) {
        int dbid = 0;
        FBlock dx{};
        std::uint32_t dhr = 0, dlr = 0;
        Code di = 0;

        if (saturated) {
            di = task;
            dbid = f_find_block(di);
            dx = D_F_BLOCK_BLOCKS[dbid];
            maskshard_split_rank(di, dx, dhr, dlr);
        } else {
            const unsigned live = __activemask();
            const int lane = int(threadIdx.x & 31);
            const int leader = __ffs(int(live)) - 1;
            const int tail = 31 - __clz(live);
            int first_bid = 0;
            int last_bid = 0;
            if (lane == leader) {
                first_bid = maskshard_loworbit_compact_find_block(task, nb);
                last_bid = maskshard_loworbit_compact_find_block(
                    task + Code(tail - leader), nb);
            }
            first_bid = __shfl_sync(live, first_bid, leader);
            last_bid = __shfl_sync(live, last_bid, leader);
            dbid = first_bid == last_bid
                ? first_bid
                : maskshard_loworbit_compact_find_block(task, nb);

            dx = D_F_BLOCK_BLOCKS[dbid];
            const Code local = task - D_MS_LOW_ORBIT_COMPACT_TASK_PREFIX[dbid];
            const std::uint32_t lc =
                D_MS_LOW_ORBIT_COMPACT_JOB_LOW_COUNT[dbid];
            if (!lc) continue;
            const std::uint32_t hq = std::uint32_t(local / Code(lc));
            const std::uint32_t lq = std::uint32_t(local - Code(hq) * Code(lc));
            const std::uint32_t ha = D_F_HIGH_MASK_OFF[
                std::size_t(D_F_MASK) * S + dx.he];
            dhr = std::uint32_t(D_MS_LOW_CLOSURE_HIGH_COMPACT_RANK[ha + hq]);
            const std::uint32_t la = D_F_LOW_ALL_OFF[dx.he];
            dlr = D_MS_LOW_ORBIT_COMPACT_LOW_RANK[la + lq];
            di = dx.off + Code(dhr) * dx.stride + dlr;
        }

        const std::size_t bdi = std::size_t(pi) * D_LOWDESC_BLOCK_TOTAL
                              + D_LOWDESC_BLOCK_BASE[dbid] + dlr;
        const std::uint32_t bdesc = D_LOWDESC_BLOCK[bdi];
        if (lowdesc_kind(bdesc) != LOWDESC_MAIN) continue;
        const std::uint32_t sbid = lowdesc_block(bdesc);
        const std::uint32_t slr = lowdesc_lr(bdesc);
        const FBlock sx = D_F_MAIN_BLOCKS[sbid];
        const Code i = sx.off + Code(dhr) * sx.stride + slr;

        const std::size_t sdi = std::size_t(pi) * D_LOWDESC_MAIN_TOTAL
                              + D_LOWDESC_MAIN_BASE[sbid] + slr;
        const std::uint32_t aux = D_MS_LOW_ORBIT_AUX[bdi];
        const std::uint32_t ak = maskshard_orbit_aux_kind(aux);
        if (ak == MS_ORBIT_AUX_INVALID) continue;

        const Count c = mainv[i];
        const Count d = blockv[di];
        if (ak == MS_ORBIT_AUX_NN || p == 1) {
            const std::uint32_t desc = D_LOWDESC_MAIN[sdi];
            if (lowdesc_kind(desc) != LOWDESC_MAIN) continue;
            const FBlock y = D_F_MAIN_BLOCKS[lowdesc_block(desc)];
            const Code j = y.off + Code(dhr) * y.stride + lowdesc_lr(desc);
            if (ak == MS_ORBIT_AUX_NN) {
                mainv[j] = maskshard_add_mod_plain(mainv[j], c);
                mainv[i] = maskshard_add_mod_plain(c, d);
                blockv[di] = 0;
            } else {
                const Count cc = mainv[j];
                mainv[i] = maskshard_add_mod_plain(
                    maskshard_add_mod_plain(c, cc), d);
                mainv[j] = maskshard_add_mod_plain(c, cc);
                blockv[di] = 0;
            }
        } else {
            const FBlock y = D_F_MAIN_BLOCKS[maskshard_orbit_aux_block(aux)];
            const Code j = y.off + Code(dhr) * y.stride
                         + maskshard_orbit_aux_rank(aux);
            const Count cc = mainv[j];
            mainv[i] = maskshard_add_mod_plain(
                maskshard_add_mod_plain(c, cc), d);
            blockv[di] = c;
        }
    }
    (void)n;
}

#ifdef maskshard_main_block_loworbit_kernel
#undef maskshard_main_block_loworbit_kernel
#endif
#define maskshard_main_block_loworbit_kernel \
        maskshard_main_block_loworbit_rowdepth_warpdecode_kernel
