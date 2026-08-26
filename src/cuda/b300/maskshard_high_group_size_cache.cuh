#pragma once

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <thread>

#ifndef MASKSHARD_HIGH_GROUP_SIZE_CACHE
#error "HIGH group-size cache requires MASKSHARD_HIGH_GROUP_SIZE_CACHE"
#endif
#ifndef MASKSHARD_HIGH_GROUP_SYNC
#error "HIGH group-size cache uses v0.57 worker-thread scoping"
#endif
#ifndef MASKSHARD_HIGH_DEAD_SYMBOL_COPIES
#error "HIGH group-size proxy requires v0.58 dead-symbol elision"
#endif

// The optimized HIGH path only needs GroupSpec::size for its FBlock consistency
// check.  The legacy configure function nevertheless rebuilds two complete
// GroupSpec DP tables for every LOW mask, row and residue.  Precompute the two
// exact sizes once per LOW mask during setup, then replace only the two later
// make_spec() call sites in the shared batch source with a tiny proxy.
//
// The one-element dp member exists solely so the legacy dead cudaMemcpyToSymbol
// expressions remain well-formed. v0.58 guarantees those symbols are elided on
// HIGH worker threads.  Calling the proxy from the main thread is rejected so a
// future code path cannot accidentally publish this intentionally incomplete DP.
struct MaskShardHighGroupSizeCache {
    std::array<Code, (1u << LOW_LUT_K)> main_size{};
    std::array<Code, (1u << LOW_LUT_K)> block_size{};
    bool built = false;

    void build() {
        if (built) return;
        constexpr std::uint32_t LM = (1u << LOW_LUT_K) - 1u;
        for (std::uint32_t mask = 0; mask < (1u << LOW_LUT_K); ++mask) {
            const GroupSpec ms = (make_spec)(TARGET_W, LM, mask);
            const GroupSpec ds = (make_spec)(TARGET_W - 1, LM, mask);
            main_size[mask] = ms.size;
            block_size[mask] = ds.size;
        }
        built = true;
        std::cerr << "HIGH group-size cache masks=" << main_size.size()
                  << " host_mib="
                  << double(sizeof(main_size) + sizeof(block_size))
                        / double(1ULL << 20)
                  << " runtime_make_spec=0\n";
    }
};

static MaskShardHighGroupSizeCache G_MS_HIGH_GROUP_SIZE_CACHE{};

// Layer setup construction onto the existing report hook while the original
// GroupSpec/make_spec identifiers are still visible.
static void maskshard_report_high_mask_shard_layout_group_size_cache(
    const MaskShardLayout& s
) {
    report_high_mask_shard_layout(s);
    G_MS_HIGH_GROUP_SIZE_CACHE.build();
}
#ifdef report_high_mask_shard_layout
#undef report_high_mask_shard_layout
#endif
#define report_high_mask_shard_layout \
        maskshard_report_high_mask_shard_layout_group_size_cache

struct MaskShardHighGroupSpecLite {
    Code size = 0;
    Code dp[1]{};
};

static MaskShardHighGroupSpecLite maskshard_high_cached_group_spec(
    int width, std::uint32_t fixed, std::uint32_t occ
) {
    constexpr std::uint32_t LM = (1u << LOW_LUT_K) - 1u;
    if (std::this_thread::get_id() == G_MS_HIGH_GROUP_SYNC_MAIN_THREAD) {
        std::cerr << "HIGH group-size proxy called on main thread\n";
        std::exit(364);
    }
    if (!G_MS_HIGH_GROUP_SIZE_CACHE.built || fixed != LM
        || occ >= G_MS_HIGH_GROUP_SIZE_CACHE.main_size.size()) {
        std::cerr << "HIGH group-size proxy invalid request width=" << width
                  << " fixed=" << fixed << " occ=" << occ << '\n';
        std::exit(365);
    }
    if (width == TARGET_W)
        return {G_MS_HIGH_GROUP_SIZE_CACHE.main_size[occ], {0}};
    if (width == TARGET_W - 1)
        return {G_MS_HIGH_GROUP_SIZE_CACHE.block_size[occ], {0}};
    std::cerr << "HIGH group-size proxy unexpected width=" << width << '\n';
    std::exit(366);
}

// The shared fullorbit-batch source has exactly two GroupSpec/make_spec uses
// after maskshard_lowclosure_kernel.cuh, both in HIGH group configuration.
#define GroupSpec MaskShardHighGroupSpecLite
#define make_spec(width, fixed, occ) \
    maskshard_high_cached_group_spec((width), (fixed), (occ))
