#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#ifndef MASKSHARD_LOW_GROUP_PACKED_CONFIG
#error "packed LOW group config header requires MASKSHARD_LOW_GROUP_PACKED_CONFIG"
#endif
#ifndef MASKSHARD_LOW_ORBIT_WARP_ROW_TASKS
#error "packed LOW group config currently layers on warp-row LOW orbit tasks"
#endif
#ifndef MASKSHARD_LOW_ORBIT_WARP_ROW_U32
#error "packed LOW group config currently requires uint32 warp-row ordinals"
#endif

// v0.38: all mutable per-LOW-group device metadata used by the active
// warp-row orbit + exact LOW closure kernels lives in one constant-memory
// object.  Older v0.37 setup issued 16 cudaMemcpyToSymbol calls per group:
// 12 base factor-group symbols, 2 compact-task symbols, and 2 warp-row symbols.
// The active kernels do not consume D_MAIN_DP/D_BLOCK_DP, fixed/occ, or
// D_F_FIX_LOW, so do not refresh those legacy symbols for LOW execution.
// HIGH execution remains unchanged and continues using the legacy D_F_* set.
struct MaskShardLowGroupPackedConfig {
    FBlock main_blocks[64]{};
    FBlock block_blocks[32]{};
    int main_nblocks = 0;
    int block_nblocks = 0;
    std::uint32_t mask = 0;
    std::uint32_t reserved = 0;
    std::uint32_t warp_prefix[HIGH_LUT_K + 3]{};
    std::uint32_t low_count[HIGH_LUT_K + 2]{};
    std::uint32_t low_chunks[HIGH_LUT_K + 2]{};
};

__device__ __constant__ MaskShardLowGroupPackedConfig D_MS_LOW_GROUP_PACKED;

static MaskShardLowGroupPackedConfig maskshard_build_low_group_packed_base(
    std::uint32_t mask
) {
    MaskShardLowGroupPackedConfig cfg{};
    const std::vector<FBlock> mb = make_factor_main_blocks(false, mask);
    const std::vector<FBlock> db = make_factor_block_blocks(false, mask);
    if (mb.empty() || db.empty() || mb.size() > 64 || db.size() > 32) {
        std::cerr << "packed LOW group invalid block counts mask=" << mask
                  << " main=" << mb.size() << " block=" << db.size() << '\n';
        std::exit(333);
    }

    Code prev = 0;
    for (std::size_t i = 0; i < mb.size(); ++i) {
        const FBlock& b = mb[i];
        if (b.off != prev || b.end < b.off) {
            std::cerr << "packed LOW group MAIN block continuity mismatch mask="
                      << mask << " bid=" << i << '\n';
            std::exit(334);
        }
        cfg.main_blocks[i] = b;
        prev = b.end;
    }
    prev = 0;
    for (std::size_t i = 0; i < db.size(); ++i) {
        const FBlock& b = db[i];
        if (b.off != prev || b.end < b.off) {
            std::cerr << "packed LOW group BLOCKED block continuity mismatch mask="
                      << mask << " bid=" << i << '\n';
            std::exit(335);
        }
        cfg.block_blocks[i] = b;
        prev = b.end;
    }
    cfg.main_nblocks = int(mb.size());
    cfg.block_nblocks = int(db.size());
    cfg.mask = mask;
    return cfg;
}

__device__ __forceinline__ int maskshard_low_packed_main_nblocks() {
    return D_MS_LOW_GROUP_PACKED.main_nblocks;
}
__device__ __forceinline__ int maskshard_low_packed_block_nblocks() {
    return D_MS_LOW_GROUP_PACKED.block_nblocks;
}
__device__ __forceinline__ FBlock maskshard_low_packed_main_block(int bid) {
    return D_MS_LOW_GROUP_PACKED.main_blocks[bid];
}
__device__ __forceinline__ FBlock maskshard_low_packed_block_block(int bid) {
    return D_MS_LOW_GROUP_PACKED.block_blocks[bid];
}
__device__ __forceinline__ std::uint32_t maskshard_low_packed_mask() {
    return D_MS_LOW_GROUP_PACKED.mask;
}
__device__ __forceinline__ std::uint32_t maskshard_low_packed_warp_prefix(int i) {
    return D_MS_LOW_GROUP_PACKED.warp_prefix[i];
}
__device__ __forceinline__ std::uint32_t maskshard_low_packed_low_count(int i) {
    return D_MS_LOW_GROUP_PACKED.low_count[i];
}
__device__ __forceinline__ std::uint32_t maskshard_low_packed_low_chunks(int i) {
    return D_MS_LOW_GROUP_PACKED.low_chunks[i];
}
