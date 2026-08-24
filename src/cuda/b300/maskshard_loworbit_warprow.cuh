#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>

#ifndef MASKSHARD_LOW_ORBIT_WARP_ROW_TASKS
#error "LOW orbit warp-row header requires MASKSHARD_LOW_ORBIT_WARP_ROW_TASKS"
#endif
#ifndef MASKSHARD_LOW_ORBIT_ROW_DEPTH_COMPACT
#error "LOW orbit warp-row tasks layer on compact row-depth tasks"
#endif

// One warp owns one active HIGH row and up to 32 active LOW ranks.  Compared
// with the flat state task map this removes per-lane 64-bit quotient/remainder
// decoding and shares HIGH-row metadata across all lanes.  At full cap, use
// physical HIGH/LOW ranks directly to preserve the saturated locality fast path.
__device__ __constant__ Code D_MS_LOW_ORBIT_WARP_PREFIX[HIGH_LUT_K + 3];
__device__ __constant__ std::uint32_t
    D_MS_LOW_ORBIT_WARP_LOW_CHUNKS[HIGH_LUT_K + 2];

static Code maskshard_configure_loworbit_warprow_plan(
    std::uint32_t mask, int cap
) {
    std::array<Code, HIGH_LUT_K + 3> state_prefix{};
    std::array<std::uint32_t, HIGH_LUT_K + 2> low_count{};
    maskshard_loworbit_rowdepth_compact_cache().make_job_plan(
        mask, cap, state_prefix, low_count);

    std::array<Code, HIGH_LUT_K + 3> warp_prefix{};
    std::array<std::uint32_t, HIGH_LUT_K + 2> low_chunks{};
    for (int h = 0; h <= HIGH_LUT_K + 1; ++h) {
        const Code states = state_prefix[size_t(h + 1)] - state_prefix[size_t(h)];
        const std::uint32_t lc = low_count[size_t(h)];
        const Code hc = lc ? states / Code(lc) : 0;
        if (lc && hc * Code(lc) != states) {
            std::cerr << "LOW orbit warp-row non-Cartesian segment mask=" << mask
                      << " h=" << h << " states=" << states
                      << " low_count=" << lc << '\n';
            std::exit(330);
        }
        const std::uint32_t chunks = (lc + 31u) >> 5;
        low_chunks[size_t(h)] = chunks;
        warp_prefix[size_t(h + 1)] =
            warp_prefix[size_t(h)] + hc * Code(chunks);
    }
    ck(cudaMemcpyToSymbol(D_MS_LOW_ORBIT_WARP_PREFIX,
                          warp_prefix.data(), sizeof(warp_prefix)),
       "LOW orbit warp-row prefix");
    ck(cudaMemcpyToSymbol(D_MS_LOW_ORBIT_WARP_LOW_CHUNKS,
                          low_chunks.data(), sizeof(low_chunks)),
       "LOW orbit warp-row LOW chunks");
    return warp_prefix[size_t(HIGH_LUT_K + 2)];
}

static void maskshard_configure_low_group_warprow(std::uint32_t mask) {
    maskshard_configure_low_group_loworbit_compact(mask);
    int dev = -1;
    ck(cudaGetDevice(&dev), "LOW orbit warp-row group get device");
    if (dev < 0 || dev >= int(G_MS_LOW_ORBIT_COMPACT_ROW.size())) {
        std::cerr << "LOW orbit warp-row unsupported device " << dev << '\n';
        std::exit(331);
    }
    const int cap = std::min(
        G_MS_LOW_ORBIT_COMPACT_ROW[size_t(dev)] + 1, TARGET_W / 2);
    (void)maskshard_configure_loworbit_warprow_plan(mask, cap);
}

#ifdef maskshard_configure_low_group
#undef maskshard_configure_low_group
#endif
#define maskshard_configure_low_group maskshard_configure_low_group_warprow

__device__ __forceinline__ int maskshard_loworbit_warprow_find_block(
    Code warp_task, int nb
) {
    int lo = 0, hi = nb + 1;
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (D_MS_LOW_ORBIT_WARP_PREFIX[mid] <= warp_task) lo = mid + 1;
        else hi = mid;
    }
    return lo - 1;
}

__global__ void maskshard_main_block_loworbit_warprow_kernel(
    Count* mainv, Count* blockv, Code n, int p
) {
    constexpr int S = MAXW + 2;
    constexpr int FULL_CAP = TARGET_W / 2;
    const int nb = D_F_BLOCK_NBLOCKS;
    if (nb <= 0) return;
    const int cap = min(D_MS_ROW_DEPTH_INDEX + 1, FULL_CAP);
    const bool saturated = cap >= FULL_CAP;
    const std::uint32_t pi = std::uint32_t(LOW_LUT_K - p);

    const unsigned active = __activemask();
    const int lane = int(threadIdx.x & 31);
    const int warp_in_block = int(threadIdx.x >> 5);
    const int warps_per_block = int((blockDim.x + 31) >> 5);
    Code warp_task = Code(blockIdx.x) * Code(warps_per_block)
                   + Code(warp_in_block);
    const Code warp_step = Code(gridDim.x) * Code(warps_per_block);
    const Code total_warps = D_MS_LOW_ORBIT_WARP_PREFIX[nb];

    for (; warp_task < total_warps; warp_task += warp_step) {
        int dbid = 0;
        std::uint32_t hq = 0;
        std::uint32_t chunk = 0;
        if (lane == 0) {
            dbid = maskshard_loworbit_warprow_find_block(warp_task, nb);
            const Code local = warp_task - D_MS_LOW_ORBIT_WARP_PREFIX[dbid];
            const std::uint32_t chunks = D_MS_LOW_ORBIT_WARP_LOW_CHUNKS[dbid];
            if (chunks) {
                hq = std::uint32_t(local / Code(chunks));
                chunk = std::uint32_t(local - Code(hq) * Code(chunks));
            }
        }
        dbid = __shfl_sync(active, dbid, 0);
        hq = __shfl_sync(active, hq, 0);
        chunk = __shfl_sync(active, chunk, 0);

        const std::uint32_t lc = D_MS_LOW_ORBIT_COMPACT_JOB_LOW_COUNT[dbid];
        const std::uint32_t lq = (chunk << 5) + std::uint32_t(lane);
        if (lq >= lc) continue;

        const FBlock dx = D_F_BLOCK_BLOCKS[dbid];
        std::uint32_t dhr = 0, dlr = 0;
        if (saturated) {
            dhr = hq;
            dlr = lq;
        } else {
            const std::uint32_t ha = D_F_HIGH_MASK_OFF[
                std::size_t(D_F_MASK) * S + dx.he];
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
        maskshard_main_block_loworbit_warprow_kernel
