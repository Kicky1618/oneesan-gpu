#pragma once

#include <algorithm>
#include <array>
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
    FBlock main_blocks[64];
    FBlock block_blocks[32];
    int main_nblocks;
    int block_nblocks;
    std::uint32_t mask;
    std::uint32_t reserved;
    std::uint32_t warp_prefix[HIGH_LUT_K + 3];
    std::uint32_t low_count[HIGH_LUT_K + 2];
    std::uint32_t low_chunks[HIGH_LUT_K + 2];
#ifdef MASKSHARD_LOW_CLOSURE_PACKED_PREFIX
    // Exact LOW closure warp-task prefix for every LOW position.  v0.41 moves
    // this from a serial per-CTA build into the existing once-per-group symbol
    // upload.  n=27 adds only 14*65*4 = 3640 bytes of constant memory.
    std::uint32_t closure_prefix[LOW_LUT_K][65];
#endif
};

__device__ __constant__ MaskShardLowGroupPackedConfig D_MS_LOW_GROUP_PACKED;

#ifdef MASKSHARD_LOW_CLOSURE_PACKED_PREFIX
#ifndef MASKSHARD_LOW_CLOSURE_ROW_DEPTH_COMPACT
#error "packed LOW closure prefix requires compact row-depth closure metadata"
#endif

// Capture only the host arrays required to build exact closure task prefixes.
// The original tables continue to be returned to the shared solver and installed
// on every GPU as before; this is host-only metadata and adds no HBM residency.
struct MaskShardLowClosurePrefixHostCapture {
    std::vector<std::uint32_t> block_off;
    std::vector<std::uint32_t> compact_active_count;
    std::vector<std::uint16_t> high_active_count;
    bool built = false;
};
static MaskShardLowClosurePrefixHostCapture G_MS_LOW_CLOSURE_PREFIX_HOST{};

static MaskShardLowClosureColsHost maskshard_build_low_closure_cols_capture(
    const StorageFactorHost& storage,
    const StorageLayout& layout,
    const LowDescHost& low_desc
) {
    MaskShardLowClosureColsHost out =
        build_maskshard_low_closure_cols(storage, layout, low_desc);
    G_MS_LOW_CLOSURE_PREFIX_HOST.block_off = out.block_off;
    G_MS_LOW_CLOSURE_PREFIX_HOST.compact_active_count = out.compact_active_count;
    G_MS_LOW_CLOSURE_PREFIX_HOST.high_active_count = out.high_active_count;
    G_MS_LOW_CLOSURE_PREFIX_HOST.built = true;
    return out;
}
#define build_maskshard_low_closure_cols maskshard_build_low_closure_cols_capture
#endif

static MaskShardLowGroupPackedConfig maskshard_build_low_group_packed_base_uncached(
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

#ifdef MASKSHARD_LOW_GROUP_PACKED_CACHE
static std::array<MaskShardLowGroupPackedConfig, (1u << HIGH_LUT_K)>
    G_MS_LOW_GROUP_PACKED_BASE_CACHE{};
static bool G_MS_LOW_GROUP_PACKED_BASE_CACHE_BUILT = false;

static void maskshard_build_low_group_packed_base_cache() {
    if (G_MS_LOW_GROUP_PACKED_BASE_CACHE_BUILT) return;
    for (std::uint32_t mask = 0; mask < (1u << HIGH_LUT_K); ++mask)
        G_MS_LOW_GROUP_PACKED_BASE_CACHE[size_t(mask)] =
            maskshard_build_low_group_packed_base_uncached(mask);
    G_MS_LOW_GROUP_PACKED_BASE_CACHE_BUILT = true;
    std::cerr << "packed LOW group base cache masks="
              << (1u << HIGH_LUT_K)
              << " mib="
              << double(sizeof(G_MS_LOW_GROUP_PACKED_BASE_CACHE))
                    / double(1ULL << 20) << '\n';
}

static MaskShardLowGroupPackedConfig maskshard_build_low_group_packed_base(
    std::uint32_t mask
) {
    if (!G_MS_LOW_GROUP_PACKED_BASE_CACHE_BUILT) {
        std::cerr << "packed LOW group base cache used before setup\n";
        std::exit(339);
    }
    if (mask >= G_MS_LOW_GROUP_PACKED_BASE_CACHE.size()) {
        std::cerr << "packed LOW group base cache invalid mask=" << mask << '\n';
        std::exit(340);
    }
    return G_MS_LOW_GROUP_PACKED_BASE_CACHE[size_t(mask)];
}

#ifdef report_high_mask_shard_layout
static void maskshard_report_high_mask_shard_layout_lowgroup_packed_cache(
    const MaskShardLayout& s
) {
    report_high_mask_shard_layout(s);
    maskshard_build_low_group_packed_base_cache();
}
#undef report_high_mask_shard_layout
#define report_high_mask_shard_layout \
        maskshard_report_high_mask_shard_layout_lowgroup_packed_cache
#else
static void maskshard_report_high_mask_shard_layout_lowgroup_packed_cache(
    const MaskShardLayout& s
) {
    report_high_mask_shard_layout(s);
    maskshard_build_low_group_packed_base_cache();
}
#define report_high_mask_shard_layout \
        maskshard_report_high_mask_shard_layout_lowgroup_packed_cache
#endif
#else
static MaskShardLowGroupPackedConfig maskshard_build_low_group_packed_base(
    std::uint32_t mask
) {
    return maskshard_build_low_group_packed_base_uncached(mask);
}
#endif

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
#ifdef MASKSHARD_LOW_CLOSURE_PACKED_PREFIX
__device__ __forceinline__ std::uint32_t maskshard_low_packed_closure_prefix(
    int pi, int i
) {
    return D_MS_LOW_GROUP_PACKED.closure_prefix[pi][i];
}
#endif
