#pragma once

#include "maskshard_low_maskbatch_config.cuh"

#ifndef MASKSHARD_LOW_MASKBATCH_REBUILD_DYNAMIC
#error "rebuild LOW mask-batch kernels require dynamic-config rebuild mode"
#endif

struct MaskShardLowBatchOrbitSharedRebuild {
    FBlock main_blocks[64];
    FBlock block_blocks[32];
    std::uint32_t warp_prefix[HIGH_LUT_K + 3];
    std::uint32_t low_count[HIGH_LUT_K + 2];
    std::uint32_t low_chunks[HIGH_LUT_K + 2];
    std::uint32_t high_mask_off[HIGH_LUT_K + 2];
};

struct MaskShardLowBatchClosureSharedRebuild {
    FBlock main_blocks[64];
    FBlock block_blocks[32];
    std::uint32_t prefix[65];
    std::uint32_t begin[65];
    std::uint32_t selected[65];
    std::uint32_t high_mask_off[HIGH_LUT_K + 2];
};

__device__ __forceinline__ int maskshard_low_batch_find_prefix_rebuild(
    const std::uint32_t* prefix,
    std::uint32_t task,
    int nb
) {
    int lo = 0, hi = nb + 1;
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (prefix[mid] <= task) lo = mid + 1;
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
    constexpr int H = HIGH_LUT_K;
    constexpr int S = FactorTablesHost::STRIDE;
    constexpr int FULL_CAP = TARGET_W / 2;
    constexpr int CAP_STRIDE = (TARGET_W + 1) / 2 + 1;
    __shared__ MaskShardLowBatchOrbitSharedRebuild s;
    (void)dynamics;

    const MaskShardLowBatchDeviceDesc job = descs[blockIdx.x];
    const MaskShardLowGroupPackedBase* base = statics + job.local;
    const int main_nb = base->main_nblocks;
    const int block_nb = base->block_nblocks;

    for (int i = int(threadIdx.x); i < main_nb; i += int(blockDim.x))
        s.main_blocks[i] = base->main_blocks[i];
    for (int i = int(threadIdx.x); i < block_nb; i += int(blockDim.x))
        s.block_blocks[i] = base->block_blocks[i];
    for (int h = int(threadIdx.x); h <= H + 1; h += int(blockDim.x))
        s.high_mask_off[h] = D_F_HIGH_MASK_OFF[
            std::size_t(job.mask) * S + std::size_t(h)];

    if (threadIdx.x == 0) {
        std::uint64_t total = 0;
        s.warp_prefix[0] = 0;
        for (int h = 0; h <= H + 1; ++h) {
            const std::uint32_t lc = D_MS_LOW_BATCH_LOW_COUNT[cap][h];
            const std::uint32_t chunks = (lc + 31u) >> 5;
            const std::uint32_t hc = std::uint32_t(
                D_MS_LOW_CLOSURE_HIGH_ACTIVE_COUNT[
                    (std::size_t(job.mask) * (H + 2) + std::size_t(h))
                        * CAP_STRIDE + std::size_t(cap)]);
            s.low_count[h] = lc;
            s.low_chunks[h] = chunks;
            total += std::uint64_t(hc) * std::uint64_t(chunks);
            s.warp_prefix[h + 1] = std::uint32_t(total);
        }
    }
    __syncthreads();

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
    const std::uint32_t total = s.warp_prefix[block_nb];

    for (; warp_task < total; warp_task += warp_step) {
        int dbid = 0;
        std::uint32_t hq = 0, chunk = 0;
        if (lane == 0) {
            dbid = maskshard_low_batch_find_prefix_rebuild(
                s.warp_prefix, warp_task, block_nb);
            const std::uint32_t local = warp_task - s.warp_prefix[dbid];
            const std::uint32_t chunks = s.low_chunks[dbid];
            if (chunks) {
                hq = local / chunks;
                chunk = local - hq * chunks;
            }
        }
        dbid = __shfl_sync(active, dbid, 0);
        hq = __shfl_sync(active, hq, 0);
        chunk = __shfl_sync(active, chunk, 0);

        const std::uint32_t lc = s.low_count[dbid];
        const std::uint32_t lq = (chunk << 5) + std::uint32_t(lane);
        if (lq >= lc) continue;

        const FBlock dx = s.block_blocks[dbid];
        std::uint32_t dhr = 0, dlr = 0;
        if (saturated) {
            dhr = hq;
            dlr = lq;
        } else {
            const std::uint32_t ha = s.high_mask_off[dx.he];
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
        const FBlock sx = s.main_blocks[sbid];
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
            const FBlock y = s.main_blocks[lowdesc_block(desc)];
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
            const FBlock y = s.main_blocks[maskshard_orbit_aux_block(aux)];
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
    constexpr int S = FactorTablesHost::STRIDE;
    constexpr int FULL_CAP = (TARGET_W + 1) / 2;
    constexpr int CAP_STRIDE = FULL_CAP + 1;
    constexpr std::uint32_t HR_MASK = (1u << H) - 1u;
    __shared__ MaskShardLowBatchClosureSharedRebuild s;
    (void)dynamics;

    const MaskShardLowBatchDeviceDesc job = descs[blockIdx.x];
    const MaskShardLowGroupPackedBase* base = statics + job.local;
    const int main_nb = base->main_nblocks;
    const int block_nb = base->block_nblocks;
    const int pi = LOW_LUT_K - p;
    const bool saturated = cap >= FULL_CAP;

    for (int i = int(threadIdx.x); i < main_nb; i += int(blockDim.x))
        s.main_blocks[i] = base->main_blocks[i];
    for (int i = int(threadIdx.x); i < block_nb; i += int(blockDim.x))
        s.block_blocks[i] = base->block_blocks[i];
    for (int h = int(threadIdx.x); h <= H + 1; h += int(blockDim.x))
        s.high_mask_off[h] = D_F_HIGH_MASK_OFF[
            std::size_t(job.mask) * S + std::size_t(h)];

    if (threadIdx.x == 0) {
        std::uint64_t total = 0;
        s.prefix[0] = 0;
        for (int b = 0; b < main_nb; ++b) {
            const FBlock x = base->main_blocks[b];
            const std::uint32_t a = D_MS_LOW_CLOSURE_BLOCK_OFF[
                std::size_t(pi) * 65 + std::size_t(b)];
            const std::uint32_t z = D_MS_LOW_CLOSURE_BLOCK_OFF[
                std::size_t(pi) * 65 + std::size_t(b + 1)];
            const std::uint32_t selected = saturated
                ? z - a
                : D_MS_LOW_CLOSURE_COMPACT_ACTIVE_COUNT[
                    (std::size_t(pi) * 65 + std::size_t(b)) * CAP_STRIDE
                    + std::size_t(cap)];
            const std::uint32_t rows = saturated
                ? (x.stride
                    ? std::uint32_t((x.end - x.off) / x.stride)
                    : 0u)
                : std::uint32_t(D_MS_LOW_CLOSURE_HIGH_ACTIVE_COUNT[
                    (std::size_t(job.mask) * (H + 2) + x.he) * CAP_STRIDE
                    + std::size_t(cap)]);
            s.begin[b] = a;
            s.selected[b] = selected;
            total += std::uint64_t(rows)
                   * std::uint64_t((selected + 31u) >> 5);
            s.prefix[b + 1] = std::uint32_t(total);
        }
    }
    __syncthreads();

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
    const std::uint32_t total = s.prefix[main_nb];

    for (; task < total; task += task_step) {
        int bid = 0;
        std::uint32_t local = 0;
        if (lane == 0) {
            bid = maskshard_low_batch_find_prefix_rebuild(s.prefix, task, main_nb);
            local = task - s.prefix[bid];
        }
        bid = __shfl_sync(active, bid, 0);
        local = __shfl_sync(active, local, 0);

        const FBlock x = s.main_blocks[bid];
        const std::uint32_t a = s.begin[bid];
        const std::uint32_t selected = s.selected[bid];
        const std::uint32_t chunks = (selected + 31u) >> 5;
        if (!chunks) continue;
        const std::uint32_t row_local = local / chunks;
        const std::uint32_t chunk = local - row_local * chunks;
        const std::uint32_t ha = s.high_mask_off[x.he];
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
            const FBlock y = s.main_blocks[lowdesc_block(desc)];
            const Code j = y.off + Code(hr) * y.stride + lowdesc_lr(desc);
            atomic_add_mod(mainv + j, c);
        } else if (kind == LOWDESC_BLOCK) {
            const FBlock y = s.block_blocks[lowdesc_block(desc)];
            const Code j = y.off + Code(hr) * y.stride + lowdesc_lr(desc);
            atomic_add_mod(blockv + j, c);
        } else if (kind == LOWDESC_CROSS) {
            const std::uint32_t hc = D_F_HIGH_MASK_CODES[ha + hr];
            const std::uint32_t hc2 = lowdesc_flip_high(hc, lowdesc_depth(desc));
            if (hc2 == 0xffffffffu) continue;
            const std::uint32_t hp = D_F_HIGH_PACKED_RANK[hc2];
            const std::uint32_t hr2 = hp & HR_MASK;
            if (p == 1) {
                const FBlock y = s.main_blocks[lowdesc_block(desc)];
                const Code j = y.off + Code(hr2) * y.stride + lowdesc_lr(desc);
                atomic_add_mod(mainv + j, c);
            } else {
                const FBlock y = s.block_blocks[lowdesc_block(desc)];
                const Code j = y.off + Code(hr2) * y.stride + lowdesc_lr(desc);
                atomic_add_mod(blockv + j, c);
            }
        }
    }
}
