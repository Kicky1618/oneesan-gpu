#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>

#ifndef MASKSHARD_LOW_GROUP_PACKED_CONFIG
#error "packed LOW warp-row header requires MASKSHARD_LOW_GROUP_PACKED_CONFIG"
#endif
#ifndef MASKSHARD_LOW_ORBIT_WARP_ROW_TASKS
#error "packed LOW warp-row header requires MASKSHARD_LOW_ORBIT_WARP_ROW_TASKS"
#endif
#ifndef MASKSHARD_LOW_ORBIT_WARP_ROW_U32
#error "packed LOW warp-row header requires uint32 warp-row tasks"
#endif
#ifndef MASKSHARD_LOW_ORBIT_ROW_DEPTH_COMPACT
#error "packed LOW warp-row header requires compact row-depth metadata"
#endif

#include "maskshard_low_group_packed_closure.cuh"

// v0.38: construct the exact state plan once, derive the warp-row plan from the
// same host arrays, and upload every mutable per-group field with one symbol
// copy.  The old compact state prefix is not copied because the active warp-row
// kernel never reads it; only its host total is retained for exact launch sizing.
static void maskshard_configure_low_group_warprow(std::uint32_t mask) {
    int dev = -1;
    ck(cudaGetDevice(&dev), "packed LOW warp-row group get device");
    if (dev < 0 || dev >= int(G_MS_LOW_ORBIT_COMPACT_ROW.size())) {
        std::cerr << "packed LOW warp-row unsupported device " << dev << '\n';
        std::exit(336);
    }

    MaskShardLowGroupPackedConfig cfg =
        maskshard_build_low_group_packed_base(mask);
    const int cap = std::min(
        G_MS_LOW_ORBIT_COMPACT_ROW[size_t(dev)] + 1, TARGET_W / 2);

    std::array<Code, HIGH_LUT_K + 3> state_prefix{};
    std::array<std::uint32_t, HIGH_LUT_K + 2> low_count{};
    const Code state_total = maskshard_loworbit_rowdepth_compact_cache().make_job_plan(
        mask, cap, state_prefix, low_count);
    G_MS_LOW_ORBIT_COMPACT_TOTAL[size_t(dev)] = state_total;

    Code warp_total = 0;
    cfg.warp_prefix[0] = 0;
    for (int h = 0; h <= HIGH_LUT_K + 1; ++h) {
        const Code states = state_prefix[size_t(h + 1)] - state_prefix[size_t(h)];
        const std::uint32_t lc = low_count[size_t(h)];
        const Code hc = lc ? states / Code(lc) : 0;
        if (lc && hc * Code(lc) != states) {
            std::cerr << "packed LOW warp-row non-Cartesian segment mask=" << mask
                      << " h=" << h << " states=" << states
                      << " low_count=" << lc << '\n';
            std::exit(337);
        }
        const std::uint32_t chunks = (lc + 31u) >> 5;
        cfg.low_count[h] = lc;
        cfg.low_chunks[h] = chunks;
        warp_total += hc * Code(chunks);
        if (warp_total > 0xffffffffULL) {
            std::cerr << "packed LOW warp-row u32 prefix overflow mask=" << mask
                      << " h=" << h << " total=" << warp_total << '\n';
            std::exit(338);
        }
        cfg.warp_prefix[h + 1] = std::uint32_t(warp_total);
    }

    ck(cudaMemcpyToSymbol(D_MS_LOW_GROUP_PACKED, &cfg, sizeof(cfg)),
       "packed LOW group config");
}

#ifdef maskshard_configure_low_group
#undef maskshard_configure_low_group
#endif
#define maskshard_configure_low_group maskshard_configure_low_group_warprow

__device__ __forceinline__ int maskshard_loworbit_warprow_packed_find_block(
    std::uint32_t warp_task, int nb
) {
    int lo = 0, hi = nb + 1;
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (maskshard_low_packed_warp_prefix(mid) <= warp_task) lo = mid + 1;
        else hi = mid;
    }
    return lo - 1;
}

__global__ void maskshard_main_block_loworbit_warprow_packed_kernel(
    Count* mainv, Count* blockv, Code n, int p
) {
    constexpr int S = MAXW + 2;
    constexpr int FULL_CAP = TARGET_W / 2;
    const int nb = maskshard_low_packed_block_nblocks();
    if (nb <= 0) return;
    const int cap = min(D_MS_ROW_DEPTH_INDEX + 1, FULL_CAP);
    const bool saturated = cap >= FULL_CAP;
    const std::uint32_t pi = std::uint32_t(LOW_LUT_K - p);
    const std::uint32_t mask = maskshard_low_packed_mask();

    const unsigned active = __activemask();
    const int lane = int(threadIdx.x & 31);
    const int warp_in_block = int(threadIdx.x >> 5);
    const int warps_per_block = int((blockDim.x + 31) >> 5);
    std::uint32_t warp_task = std::uint32_t(blockIdx.x) * std::uint32_t(warps_per_block)
                            + std::uint32_t(warp_in_block);
    const std::uint32_t warp_step =
        std::uint32_t(gridDim.x) * std::uint32_t(warps_per_block);
    const std::uint32_t total_warps = maskshard_low_packed_warp_prefix(nb);

    for (; warp_task < total_warps; warp_task += warp_step) {
        int dbid = 0;
        std::uint32_t hq = 0;
        std::uint32_t chunk = 0;
        if (lane == 0) {
            dbid = maskshard_loworbit_warprow_packed_find_block(warp_task, nb);
            const std::uint32_t local =
                warp_task - maskshard_low_packed_warp_prefix(dbid);
            const std::uint32_t chunks = maskshard_low_packed_low_chunks(dbid);
            if (chunks) {
                hq = local / chunks;
                chunk = local - hq * chunks;
            }
        }
        dbid = __shfl_sync(active, dbid, 0);
        hq = __shfl_sync(active, hq, 0);
        chunk = __shfl_sync(active, chunk, 0);

        const std::uint32_t lc = maskshard_low_packed_low_count(dbid);
        const std::uint32_t lq = (chunk << 5) + std::uint32_t(lane);
        if (lq >= lc) continue;

        const FBlock dx = maskshard_low_packed_block_block(dbid);
        std::uint32_t dhr = 0, dlr = 0;
        if (saturated) {
            dhr = hq;
            dlr = lq;
        } else {
            const std::uint32_t ha = D_F_HIGH_MASK_OFF[
                std::size_t(mask) * S + dx.he];
            dhr = std::uint32_t(D_MS_LOW_CLOSURE_HIGH_COMPACT_RANK[ha + hq]);
            const std::uint32_t la = D_F_LOW_ALL_OFF[dx.he];
            dlr = D_MS_LOW_ORBIT_COMPACT_LOW_RANK[la + lq];
        }
        const Code di = dx.off + Code(dhr) * dx.stride + dlr;

        const std::size_t bdi = std::size_t(pi) * D_LOWDESC_BLOCK_TOTAL
                              + D_LOWDESC_BLOCK_BASE[dbid] + dlr;
        const std::uint32_t bdesc = D_LOWDESC_BLOCK[bdi];
        if (lowdesc_kind(bdesc) != LOWDESC_MAIN) continue;
        const std::uint32_t sbid = lowdesc_block(bdesc);
        const std::uint32_t slr = lowdesc_lr(bdesc);
        const FBlock sx = maskshard_low_packed_main_block(sbid);
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
            const FBlock y = maskshard_low_packed_main_block(lowdesc_block(desc));
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
            const FBlock y = maskshard_low_packed_main_block(
                maskshard_orbit_aux_block(aux));
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
        maskshard_main_block_loworbit_warprow_packed_kernel
