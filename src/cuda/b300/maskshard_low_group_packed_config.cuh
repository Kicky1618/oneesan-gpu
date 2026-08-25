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

struct MaskShardLowGroupPackedBase {
    FBlock main_blocks[64];
    FBlock block_blocks[32];
    int main_nblocks;
    int block_nblocks;
    std::uint32_t mask;
    std::uint32_t reserved;
};

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
    std::uint32_t closure_prefix[LOW_LUT_K][65];
#endif
#ifdef MASKSHARD_LOW_CLOSURE_PACKED_META
    std::uint32_t closure_begin[LOW_LUT_K][65];
    std::uint32_t closure_selected[LOW_LUT_K][65];
    std::uint32_t high_mask_off[HIGH_LUT_K + 2];
#endif
};

__device__ __constant__ MaskShardLowGroupPackedConfig D_MS_LOW_GROUP_PACKED;

#ifdef MASKSHARD_LOW_CLOSURE_PACKED_PREFIX
#ifndef MASKSHARD_LOW_CLOSURE_ROW_DEPTH_COMPACT
#error "packed LOW closure prefix requires compact row-depth closure metadata"
#endif
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

static MaskShardLowGroupPackedBase maskshard_build_low_group_packed_static_uncached(
    std::uint32_t mask
) {
    MaskShardLowGroupPackedBase base{};
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
        base.main_blocks[i] = b;
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
        base.block_blocks[i] = b;
        prev = b.end;
    }
    base.main_nblocks = int(mb.size());
    base.block_nblocks = int(db.size());
    base.mask = mask;
    return base;
}

static void maskshard_apply_low_group_packed_base(
    MaskShardLowGroupPackedConfig& cfg,
    const MaskShardLowGroupPackedBase& base
) {
    for (int i = 0; i < 64; ++i) cfg.main_blocks[i] = base.main_blocks[i];
    for (int i = 0; i < 32; ++i) cfg.block_blocks[i] = base.block_blocks[i];
    cfg.main_nblocks = base.main_nblocks;
    cfg.block_nblocks = base.block_nblocks;
    cfg.mask = base.mask;
    cfg.reserved = 0;
}

static MaskShardLowGroupPackedConfig maskshard_build_low_group_packed_base_uncached(
    std::uint32_t mask
) {
    MaskShardLowGroupPackedConfig cfg{};
    maskshard_apply_low_group_packed_base(
        cfg, maskshard_build_low_group_packed_static_uncached(mask));
    return cfg;
}

#ifdef MASKSHARD_LOW_GROUP_PACKED_CACHE
#ifdef MASKSHARD_LOW_GROUP_STATIC_BASE_CACHE
static std::array<MaskShardLowGroupPackedBase, (1u << HIGH_LUT_K)>
    G_MS_LOW_GROUP_PACKED_BASE_CACHE{};
#else
static std::array<MaskShardLowGroupPackedConfig, (1u << HIGH_LUT_K)>
    G_MS_LOW_GROUP_PACKED_BASE_CACHE{};
#endif
static bool G_MS_LOW_GROUP_PACKED_BASE_CACHE_BUILT = false;

static void maskshard_build_low_group_packed_base_cache() {
    if (G_MS_LOW_GROUP_PACKED_BASE_CACHE_BUILT) return;
    for (std::uint32_t mask = 0; mask < (1u << HIGH_LUT_K); ++mask) {
#ifdef MASKSHARD_LOW_GROUP_STATIC_BASE_CACHE
        G_MS_LOW_GROUP_PACKED_BASE_CACHE[size_t(mask)] =
            maskshard_build_low_group_packed_static_uncached(mask);
#else
        G_MS_LOW_GROUP_PACKED_BASE_CACHE[size_t(mask)] =
            maskshard_build_low_group_packed_base_uncached(mask);
#endif
    }
    G_MS_LOW_GROUP_PACKED_BASE_CACHE_BUILT = true;
    std::cerr << "packed LOW group base cache masks="
              << (1u << HIGH_LUT_K)
#ifdef MASKSHARD_LOW_GROUP_STATIC_BASE_CACHE
              << " static_only=1 base_bytes=" << sizeof(MaskShardLowGroupPackedBase)
#else
              << " static_only=0 base_bytes=" << sizeof(MaskShardLowGroupPackedConfig)
#endif
              << " config_bytes=" << sizeof(MaskShardLowGroupPackedConfig)
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
#ifdef MASKSHARD_LOW_GROUP_STATIC_BASE_CACHE
    MaskShardLowGroupPackedConfig cfg{};
    maskshard_apply_low_group_packed_base(
        cfg, G_MS_LOW_GROUP_PACKED_BASE_CACHE[size_t(mask)]);
    return cfg;
#else
    return G_MS_LOW_GROUP_PACKED_BASE_CACHE[size_t(mask)];
#endif
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
#ifdef MASKSHARD_LOW_CLOSURE_PACKED_META
__device__ __forceinline__ std::uint32_t maskshard_low_packed_closure_begin(
    int pi, int bid
) {
    return D_MS_LOW_GROUP_PACKED.closure_begin[pi][bid];
}
__device__ __forceinline__ std::uint32_t maskshard_low_packed_closure_selected(
    int pi, int bid
) {
    return D_MS_LOW_GROUP_PACKED.closure_selected[pi][bid];
}
__device__ __forceinline__ std::uint32_t maskshard_low_packed_high_mask_off(int h) {
    return D_MS_LOW_GROUP_PACKED.high_mask_off[h];
}
#endif
