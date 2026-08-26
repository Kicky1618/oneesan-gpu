#pragma once

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <utility>
#include <vector>

#ifndef MASKSHARD_HIGH_FBLOCK_CACHE
#error "HIGH FBlock cache requires MASKSHARD_HIGH_FBLOCK_CACHE"
#endif
#ifndef MASKSHARD_HIGH_GROUP_SYNC
#error "HIGH FBlock cache uses v0.57 worker-thread scoping"
#endif

// HIGH jobs fix the complete LOW occupancy mask.  Their factorized MAIN/BLOCKED
// FBlock layouts depend only on that LOW mask, yet the legacy host path rebuilds
// both std::vector<FBlock> objects for every row and residue.  build_high_jobs()
// already visits every LOW mask once on the main thread before workers start, so
// retain those first vectors and return allocation-free views thereafter.
//
// Calls with fix_low=false belong to the LOW path and are forwarded to the
// original builders unchanged.  The view has an rvalue conversion back to
// std::vector<FBlock> so older LOW helper code with an explicit vector type keeps
// exactly its previous behavior.
class MaskShardHighFBlockView {
    const std::vector<FBlock>* cached_ = nullptr;
    std::vector<FBlock> owned_;

    const std::vector<FBlock>& vec() const {
        return cached_ ? *cached_ : owned_;
    }

public:
    MaskShardHighFBlockView() = default;
    explicit MaskShardHighFBlockView(const std::vector<FBlock>* cached)
        : cached_(cached) {}
    explicit MaskShardHighFBlockView(std::vector<FBlock>&& owned)
        : owned_(std::move(owned)) {}

    bool empty() const { return vec().empty(); }
    std::size_t size() const { return vec().size(); }
    const FBlock& back() const { return vec().back(); }
    const FBlock* data() const { return vec().data(); }
    const FBlock& operator[](std::size_t i) const { return vec()[i]; }
    auto begin() const { return vec().begin(); }
    auto end() const { return vec().end(); }

    operator std::vector<FBlock>() && {
        if (cached_) return *cached_;
        return std::move(owned_);
    }
    operator std::vector<FBlock>() const & { return vec(); }
};

struct MaskShardHighFBlockCacheEntry {
    std::vector<FBlock> main_blocks;
    std::vector<FBlock> block_blocks;
    bool main_built = false;
    bool block_built = false;
};

static std::array<MaskShardHighFBlockCacheEntry, (1u << LOW_LUT_K)>
    G_MS_HIGH_FBLOCK_CACHE{};

static MaskShardHighFBlockView maskshard_high_cached_main_blocks(
    bool fix_low, std::uint32_t mask
) {
    if (!fix_low)
        return MaskShardHighFBlockView(
            (make_factor_main_blocks)(false, mask));
    if (mask >= G_MS_HIGH_FBLOCK_CACHE.size()) {
        std::cerr << "HIGH FBlock MAIN cache mask overflow mask=" << mask << '\n';
        std::exit(360);
    }
    auto& e = G_MS_HIGH_FBLOCK_CACHE[mask];
    if (!e.main_built) {
        if (std::this_thread::get_id() != G_MS_HIGH_GROUP_SYNC_MAIN_THREAD) {
            std::cerr << "HIGH FBlock MAIN cache miss on worker mask=" << mask << '\n';
            std::exit(361);
        }
        e.main_blocks = (make_factor_main_blocks)(true, mask);
        e.main_built = true;
    }
    return MaskShardHighFBlockView(&e.main_blocks);
}

static MaskShardHighFBlockView maskshard_high_cached_block_blocks(
    bool fix_low, std::uint32_t mask
) {
    if (!fix_low)
        return MaskShardHighFBlockView(
            (make_factor_block_blocks)(false, mask));
    if (mask >= G_MS_HIGH_FBLOCK_CACHE.size()) {
        std::cerr << "HIGH FBlock BLOCKED cache mask overflow mask=" << mask << '\n';
        std::exit(362);
    }
    auto& e = G_MS_HIGH_FBLOCK_CACHE[mask];
    if (!e.block_built) {
        if (std::this_thread::get_id() != G_MS_HIGH_GROUP_SYNC_MAIN_THREAD) {
            std::cerr << "HIGH FBlock BLOCKED cache miss on worker mask=" << mask << '\n';
            std::exit(363);
        }
        e.block_blocks = (make_factor_block_blocks)(true, mask);
        e.block_built = true;
        if (mask + 1 == G_MS_HIGH_FBLOCK_CACHE.size()) {
            std::size_t payload = 0;
            for (const auto& x : G_MS_HIGH_FBLOCK_CACHE)
                payload += (x.main_blocks.size() + x.block_blocks.size())
                         * sizeof(FBlock);
            std::cerr << "HIGH FBlock cache masks="
                      << G_MS_HIGH_FBLOCK_CACHE.size()
                      << " payload_mib="
                      << double(payload) / double(1ULL << 20)
                      << " worker_rebuilds=0\n";
        }
    }
    return MaskShardHighFBlockView(&e.block_blocks);
}

// Function-like macros do not expand the parenthesized original names used in
// the wrappers above.
#define make_factor_main_blocks(fix_low, mask) \
    maskshard_high_cached_main_blocks((fix_low), (mask))
#define make_factor_block_blocks(fix_low, mask) \
    maskshard_high_cached_block_blocks((fix_low), (mask))
