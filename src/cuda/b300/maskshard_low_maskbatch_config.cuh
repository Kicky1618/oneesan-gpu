#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#ifndef MASKSHARD_LOW_CLOSURE_PACKED_PREFIX
#error "resident LOW mask-batch configs require prepacked closure prefixes"
#endif
#ifndef MASKSHARD_LOW_CLOSURE_PACKED_META
#error "resident LOW mask-batch configs require packed closure metadata"
#endif
#ifndef MASKSHARD_LOW_GROUP_STATIC_BASE_CACHE
#error "resident LOW mask-batch configs require static-only base cache"
#endif

#include "maskshard_low_maskbatch_rowplan.hpp"

static constexpr int MASKSHARD_LOW_BATCH_ROW_CAPS = (TARGET_W + 1) / 2;

#ifdef MASKSHARD_LOW_MASKBATCH_REBUILD_DYNAMIC
// v0.48: LOW-all active counts are mask-independent.  Publish the tiny
// [cap][ending-height] table once per device; CTA kernels combine it with the
// already-resident HIGH active-count table to rebuild exact orbit prefixes.
__device__ __constant__ std::uint32_t
    D_MS_LOW_BATCH_LOW_COUNT[MASKSHARD_LOW_BATCH_ROW_CAPS + 1][HIGH_LUT_K + 2];
#endif

struct MaskShardLowBatchDynamicConfig {
#ifdef MASKSHARD_LOW_MASKBATCH_REBUILD_DYNAMIC
    std::uint32_t unused = 0;
#else
    std::uint32_t warp_prefix[HIGH_LUT_K + 3];
    std::uint32_t low_count[HIGH_LUT_K + 2];
    std::uint32_t low_chunks[HIGH_LUT_K + 2];
    std::uint32_t closure_prefix[LOW_LUT_K][65];
#ifndef MASKSHARD_LOW_MASKBATCH_COMPACT_DYNAMIC
    // v0.43-v0.46 duplicate these mask-independent arrays for every mask/cap.
    // v0.47 CTA-cached kernels stage them directly from the canonical LOW
    // closure device tables instead, removing 2*LOW_LUT_K*65 uint32 words.
    std::uint32_t closure_begin[LOW_LUT_K][65];
    std::uint32_t closure_selected[LOW_LUT_K][65];
#endif
    std::uint32_t high_mask_off[HIGH_LUT_K + 2];
#endif
};

struct MaskShardLowBatchDeviceDesc {
    std::uint16_t mask = 0;
    std::uint16_t local = 0;
    std::uint8_t replica = 0;
    std::uint8_t replicas = 0;
    std::uint16_t reserved = 0;
};
static_assert(sizeof(MaskShardLowBatchDeviceDesc) == 8,
              "LOW mask-batch device descriptor ABI changed");

#ifndef MASKSHARD_LOW_MASKBATCH_REBUILD_DYNAMIC
static MaskShardLowBatchDynamicConfig maskshard_extract_low_batch_dynamic(
    const MaskShardLowGroupPackedConfig& cfg
) {
    MaskShardLowBatchDynamicConfig d{};
    for (int i = 0; i < HIGH_LUT_K + 3; ++i)
        d.warp_prefix[i] = cfg.warp_prefix[i];
    for (int i = 0; i < HIGH_LUT_K + 2; ++i) {
        d.low_count[i] = cfg.low_count[i];
        d.low_chunks[i] = cfg.low_chunks[i];
        d.high_mask_off[i] = cfg.high_mask_off[i];
    }
    for (int pi = 0; pi < LOW_LUT_K; ++pi)
        for (int b = 0; b < 65; ++b) {
            d.closure_prefix[pi][b] = cfg.closure_prefix[pi][b];
#ifndef MASKSHARD_LOW_MASKBATCH_COMPACT_DYNAMIC
            d.closure_begin[pi][b] = cfg.closure_begin[pi][b];
            d.closure_selected[pi][b] = cfg.closure_selected[pi][b];
#endif
        }
    return d;
}
#endif

struct MaskShardLowMaskBatchDeviceTables {
    static constexpr int FULL_CAP = MASKSHARD_LOW_BATCH_ROW_CAPS;

    int dev = -1;
    std::vector<std::uint16_t> local_of_mask;
    std::vector<std::uint16_t> mask_of_local;
    MaskShardLowGroupPackedBase* d_static = nullptr;
    MaskShardLowBatchDynamicConfig* d_dynamic = nullptr;

    void install(
        int device,
        const MaskShardLayout& shard,
        const MaskShardLowMaskBatchTablesHost& tasks
    ) {
        dev = device;
        local_of_mask.assign(shard.masks, std::uint16_t(0xffffu));
        for (std::uint32_t mask = 0; mask < shard.masks; ++mask) {
            if (int(shard.owner[mask]) != dev) continue;
            if (mask_of_local.size() >= 0xffffu) {
                std::cerr << "LOW mask-batch local mask index overflow dev="
                          << dev << '\n';
                std::exit(349);
            }
            local_of_mask[mask] = std::uint16_t(mask_of_local.size());
            mask_of_local.push_back(std::uint16_t(mask));
        }

#ifdef MASKSHARD_LOW_MASKBATCH_REBUILD_DYNAMIC
        {
            auto& orbit = maskshard_loworbit_rowdepth_compact_cache();
            orbit.build();
            std::array<std::uint32_t,
                       (FULL_CAP + 1) * (HIGH_LUT_K + 2)> hlow{};
            for (int cap = 1; cap <= FULL_CAP; ++cap) {
                const int orbit_cap = std::min(cap, TARGET_W / 2);
                for (int h = 0; h <= HIGH_LUT_K + 1; ++h)
                    hlow[std::size_t(cap) * (HIGH_LUT_K + 2) + std::size_t(h)] =
                        orbit.low_count[std::size_t(h)][std::size_t(orbit_cap)];
            }
            ck(cudaMemcpyToSymbol(D_MS_LOW_BATCH_LOW_COUNT,
                                  hlow.data(), sizeof(hlow)),
               "LOW mask-batch LOW-count table");
        }
#endif

        std::vector<MaskShardLowGroupPackedBase> hs(mask_of_local.size());
#ifndef MASKSHARD_LOW_MASKBATCH_REBUILD_DYNAMIC
        std::vector<MaskShardLowBatchDynamicConfig> hd(
            mask_of_local.size() * FULL_CAP);
#endif
        for (std::size_t local = 0; local < mask_of_local.size(); ++local) {
            const std::uint32_t mask = mask_of_local[local];
            hs[local] = maskshard_build_low_group_packed_static_uncached(mask);
            for (int cap = 1; cap <= FULL_CAP; ++cap) {
                Code state_total = 0;
                const MaskShardLowGroupPackedConfig cfg =
                    maskshard_build_low_group_packed_config(
                        mask, cap - 1, &state_total);
                const std::uint32_t orbit_tasks =
                    cfg.warp_prefix[cfg.block_nblocks];
                if (orbit_tasks != tasks.orbit(mask, cap)) {
                    std::cerr << "LOW mask-batch orbit config/task mismatch dev="
                              << dev << " mask=" << mask << " cap=" << cap
                              << " got=" << orbit_tasks
                              << " expected=" << tasks.orbit(mask, cap) << '\n';
                    std::exit(350);
                }
                for (int pi = 0; pi < LOW_LUT_K; ++pi) {
                    const std::uint32_t z = cfg.closure_prefix[pi][cfg.main_nblocks];
                    if (z != tasks.closure(mask, cap, pi)) {
                        std::cerr << "LOW mask-batch closure config/task mismatch dev="
                                  << dev << " mask=" << mask << " cap=" << cap
                                  << " pi=" << pi << " got=" << z
                                  << " expected=" << tasks.closure(mask, cap, pi)
                                  << '\n';
                        std::exit(351);
                    }
                }
#ifndef MASKSHARD_LOW_MASKBATCH_REBUILD_DYNAMIC
                hd[local * FULL_CAP + std::size_t(cap - 1)] =
                    maskshard_extract_low_batch_dynamic(cfg);
#endif
            }
        }

        if (!hs.empty()) {
            ck(cudaMalloc(&d_static, hs.size() * sizeof(hs[0])),
               "LOW mask-batch static config alloc");
            ck(cudaMemcpy(d_static, hs.data(), hs.size() * sizeof(hs[0]),
                          cudaMemcpyHostToDevice),
               "LOW mask-batch static config copy");
        }
#ifndef MASKSHARD_LOW_MASKBATCH_REBUILD_DYNAMIC
        if (!hd.empty()) {
            ck(cudaMalloc(&d_dynamic, hd.size() * sizeof(hd[0])),
               "LOW mask-batch dynamic config alloc");
            ck(cudaMemcpy(d_dynamic, hd.data(), hd.size() * sizeof(hd[0]),
                          cudaMemcpyHostToDevice),
               "LOW mask-batch dynamic config copy");
        }
#endif

        std::cerr << "LOW mask-batch resident config dev=" << dev
                  << " masks=" << mask_of_local.size()
                  << " static_mib="
                  << double(hs.size() * sizeof(hs[0])) / double(1ULL << 20)
#ifdef MASKSHARD_LOW_MASKBATCH_REBUILD_DYNAMIC
                  << " dynamic_bytes_per_mask_cap=0 dynamic_mib=0"
                  << " rebuild_dynamic=1"
#else
                  << " dynamic_bytes_per_mask_cap="
                  << sizeof(MaskShardLowBatchDynamicConfig)
                  << " dynamic_mib="
                  << double(hd.size() * sizeof(hd[0])) / double(1ULL << 20)
#ifdef MASKSHARD_LOW_MASKBATCH_COMPACT_DYNAMIC
                  << " compact_dynamic=1"
#else
                  << " compact_dynamic=0"
#endif
#endif
                  << '\n';
    }

    std::vector<MaskShardLowBatchDeviceDesc> bind(
        const std::vector<MaskShardLowBatchDesc>& src
    ) const {
        std::vector<MaskShardLowBatchDeviceDesc> out;
        out.reserve(src.size());
        for (const MaskShardLowBatchDesc& x : src) {
            if (x.mask >= local_of_mask.size()
                || local_of_mask[x.mask] == std::uint16_t(0xffffu)) {
                std::cerr << "LOW mask-batch descriptor owner mismatch dev="
                          << dev << " mask=" << x.mask << '\n';
                std::exit(352);
            }
            out.push_back({x.mask, local_of_mask[x.mask],
                           x.replica, x.replicas, 0});
        }
        return out;
    }

    void release() {
        if (dev >= 0) cudaSetDevice(dev);
        if (d_static) cudaFree(d_static);
        if (d_dynamic) cudaFree(d_dynamic);
        d_static = nullptr;
        d_dynamic = nullptr;
        local_of_mask.clear();
        mask_of_local.clear();
    }
};
