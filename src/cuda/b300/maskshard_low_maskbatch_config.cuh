#pragma once

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

struct MaskShardLowBatchDynamicConfig {
    std::uint32_t warp_prefix[HIGH_LUT_K + 3];
    std::uint32_t low_count[HIGH_LUT_K + 2];
    std::uint32_t low_chunks[HIGH_LUT_K + 2];
    std::uint32_t closure_prefix[LOW_LUT_K][65];
    std::uint32_t closure_begin[LOW_LUT_K][65];
    std::uint32_t closure_selected[LOW_LUT_K][65];
    std::uint32_t high_mask_off[HIGH_LUT_K + 2];
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
            d.closure_begin[pi][b] = cfg.closure_begin[pi][b];
            d.closure_selected[pi][b] = cfg.closure_selected[pi][b];
        }
    return d;
}

struct MaskShardLowMaskBatchDeviceTables {
    static constexpr int FULL_CAP = (TARGET_W + 1) / 2;

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

        std::vector<MaskShardLowGroupPackedBase> hs(mask_of_local.size());
        std::vector<MaskShardLowBatchDynamicConfig> hd(
            mask_of_local.size() * FULL_CAP);
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
                hd[local * FULL_CAP + std::size_t(cap - 1)] =
                    maskshard_extract_low_batch_dynamic(cfg);
            }
        }

        if (!hs.empty()) {
            ck(cudaMalloc(&d_static, hs.size() * sizeof(hs[0])),
               "LOW mask-batch static config alloc");
            ck(cudaMemcpy(d_static, hs.data(), hs.size() * sizeof(hs[0]),
                          cudaMemcpyHostToDevice),
               "LOW mask-batch static config copy");
        }
        if (!hd.empty()) {
            ck(cudaMalloc(&d_dynamic, hd.size() * sizeof(hd[0])),
               "LOW mask-batch dynamic config alloc");
            ck(cudaMemcpy(d_dynamic, hd.data(), hd.size() * sizeof(hd[0]),
                          cudaMemcpyHostToDevice),
               "LOW mask-batch dynamic config copy");
        }

        std::cerr << "LOW mask-batch resident config dev=" << dev
                  << " masks=" << mask_of_local.size()
                  << " static_mib="
                  << double(hs.size() * sizeof(hs[0])) / double(1ULL << 20)
                  << " dynamic_mib="
                  << double(hd.size() * sizeof(hd[0])) / double(1ULL << 20)
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
