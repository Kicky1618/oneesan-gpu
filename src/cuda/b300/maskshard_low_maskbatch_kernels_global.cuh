#pragma once

#include "maskshard_low_maskbatch_config.cuh"

__device__ __forceinline__ int maskshard_low_batch_find_orbit_block(
    const MaskShardLowBatchDynamicConfig& dyn,
    std::uint32_t task,
    int nb
) {
    int lo = 0, hi = nb + 1;
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (dyn.warp_prefix[mid] <= task) lo = mid + 1;
        else hi = mid;
    }
    return lo - 1;
}

__device__ __forceinline__ int maskshard_low_batch_find_closure_block(
    const MaskShardLowBatchDynamicConfig& dyn,
    int pi,
    std::uint32_t task,
    int nb
) {
    int lo = 0, hi = nb + 1;
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (dyn.closure_prefix[pi][mid] <= task) lo = mid + 1;
        else hi = mid;
    }
    return lo - 1;
}

__global__ void maskshard_low_maskbatch_orbit_kernel(
    Count* authoritative_main,
    Count* authoritative_block,
    const MaskShardLowBatchDeviceDesc* descs,
    const MaskShardLowGroupPackedBase* statics,
    const MaskShardLowBatchDynamicConfig* dynamics,
    int cap,
    int p
) {
    constexpr int FULL_CAP = TARGET_W / 2;
    const MaskShardLowBatchDeviceDesc job = descs[blockIdx.x];
    const MaskShardLowGroupPackedBase* base = statics + job.local;
    const MaskShardLowBatchDynamicConfig& dyn =
        dynamics[std::size_t(job.local) * ((TARGET_W + 1) / 2)
                 + std::size_t(cap - 1)];
    const bool saturated = cap >= FULL_CAP;
    const std::uint32_t pi = std::uint32_t(LOW_LUT_K - p);
    Count* mainv = authoritative_main + D_MS_MAIN_BASE[job.mask];
    Count* blockv = authoritative_block + D_MS_BLOCK_BASE[job.mask];

    const unsigned active = __activemask();
    const int lane = int(threadIdx.x & 31);
    const int warp_in_block = int(threadIdx.x >> 5);
    const std::uint32_t warps_per_block =
        std::uint32_t((blockDim.x + 31) >> 5);
    std::uint32_t warp_task = std::uint32_t(job.replica) * warps_per_block
                            + std::uint32_t(warp_in_block);
    const std::uint32_t warp_step =
        std::uint32_t(job.replicas) * warps_per_block;
    const int nb = base->block_nblocks;
    const std::uint32_t total = dyn.warp_prefix[nb];

    for (; warp_task < total; warp_task += warp_step) {
        int dbid = 0;
        std::uint32_t hq = 0, chunk = 0;
        if (lane == 0) {
            dbid = maskshard_low_batch_find_orbit_block(dyn, warp_task, nb);
            const std::uint32_t local = warp_task - dyn.warp_prefix[dbid];
            const std::uint32_t chunks = dyn.low_chunks[dbid];
            if (chunks) {
                hq = local / chunks;
                chunk = local - hq * chunks;
            }
        }
        dbid = __shfl_sync(active, dbid, 0);
        hq = __shfl_sync(active, hq, 0);
        chunk = __shfl_sync(active, chunk, 0);

        const std::uint32_t lc = dyn.low_count[dbid];
        const std::uint32_t lq = (chunk << 5) + std::uint32_t(lane);
        if (lq >= lc) continue;

        const FBlock dx = base->block_blocks[dbid];
        std::uint32_t dhr = 0, dlr = 0;
        if (saturated) {
            dhr = hq;
            dlr = lq;
        } else {
            const std::uint32_t ha = dyn.high_mask_off[dx.he];
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
        const FBlock sx = base->main_blocks[sbid];
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
            const FBlock y = base->main_blocks[lowdesc_block(desc)];
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
            const FBlock y = base->main_blocks[maskshard_orbit_aux_block(aux)];
            const Code j = y.off + Code(dhr) * y.stride
                         + maskshard_orbit_aux_rank(aux);
            const Count cc = mainv[j];
            mainv[i] = maskshard_add_mod_plain(
                maskshard_add_mod_plain(c, cc), d);
            blockv[di] = c;
        }
    }
}

__global__ void maskshard_low_maskbatch_closure_kernel(
    Count* authoritative_main,
    Count* authoritative_block,
    const MaskShardLowBatchDeviceDesc* descs,
    const MaskShardLowGroupPackedBase* statics,
    const MaskShardLowBatchDynamicConfig* dynamics,
    int cap,
    int p
) {
    constexpr int H = HIGH_LUT_K;
    constexpr int FULL_CAP = (TARGET_W + 1) / 2;
    constexpr std::uint32_t HR_MASK = (1u << H) - 1u;
    const MaskShardLowBatchDeviceDesc job = descs[blockIdx.x];
    const MaskShardLowGroupPackedBase* base = statics + job.local;
    const MaskShardLowBatchDynamicConfig& dyn =
        dynamics[std::size_t(job.local) * FULL_CAP + std::size_t(cap - 1)];
    const bool saturated = cap >= FULL_CAP;
    const int pi = LOW_LUT_K - p;
    Count* mainv = authoritative_main + D_MS_MAIN_BASE[job.mask];
    Count* blockv = authoritative_block + D_MS_BLOCK_BASE[job.mask];

    const unsigned active = __activemask();
    const int lane = int(threadIdx.x & 31);
    const int warp_in_block = int(threadIdx.x >> 5);
    const std::uint32_t warps_per_block =
        std::uint32_t((blockDim.x + 31) >> 5);
    std::uint32_t task = std::uint32_t(job.replica) * warps_per_block
                       + std::uint32_t(warp_in_block);
    const std::uint32_t task_step =
        std::uint32_t(job.replicas) * warps_per_block;
    const int nb = base->main_nblocks;
    const std::uint32_t total = dyn.closure_prefix[pi][nb];

    for (; task < total; task += task_step) {
        int bid = 0;
        std::uint32_t local = 0;
        if (lane == 0) {
            bid = maskshard_low_batch_find_closure_block(dyn, pi, task, nb);
            local = task - dyn.closure_prefix[pi][bid];
        }
        bid = __shfl_sync(active, bid, 0);
        local = __shfl_sync(active, local, 0);

        const FBlock x = base->main_blocks[bid];
        const std::uint32_t a = dyn.closure_begin[pi][bid];
        const std::uint32_t selected = dyn.closure_selected[pi][bid];
        const std::uint32_t chunks = (selected + 31u) >> 5;
        if (!chunks) continue;
        const std::uint32_t row_local = local / chunks;
        const std::uint32_t chunk = local - row_local * chunks;
        const std::uint32_t ha = dyn.high_mask_off[x.he];
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
            const FBlock y = base->main_blocks[lowdesc_block(desc)];
            const Code j = y.off + Code(hr) * y.stride + lowdesc_lr(desc);
            atomic_add_mod(mainv + j, c);
        } else if (kind == LOWDESC_BLOCK) {
            const FBlock y = base->block_blocks[lowdesc_block(desc)];
            const Code j = y.off + Code(hr) * y.stride + lowdesc_lr(desc);
            atomic_add_mod(blockv + j, c);
        } else if (kind == LOWDESC_CROSS) {
            const std::uint32_t hc = D_F_HIGH_MASK_CODES[ha + hr];
            const std::uint32_t hc2 = lowdesc_flip_high(hc, lowdesc_depth(desc));
            if (hc2 == 0xffffffffu) continue;
            const std::uint32_t hp = D_F_HIGH_PACKED_RANK[hc2];
            const std::uint32_t hr2 = hp & HR_MASK;
            if (p == 1) {
                const FBlock y = base->main_blocks[lowdesc_block(desc)];
                const Code j = y.off + Code(hr2) * y.stride + lowdesc_lr(desc);
                atomic_add_mod(mainv + j, c);
            } else {
                const FBlock y = base->block_blocks[lowdesc_block(desc)];
                const Code j = y.off + Code(hr2) * y.stride + lowdesc_lr(desc);
                atomic_add_mod(blockv + j, c);
            }
        }
    }
}
