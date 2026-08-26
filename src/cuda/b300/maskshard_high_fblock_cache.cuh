#pragma once

#include <algorithm>
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

// HIGH jobs fix the complete LOW occupancy mask. Their factorized MAIN/BLOCKED
// FBlock layouts depend only on that LOW mask, yet the legacy host path rebuilds
// both std::vector<FBlock> objects for every row and residue. build_high_jobs()
// already visits every LOW mask once on the main thread before workers start, so
// retain those first layouts and return allocation-free views thereafter.
//
// Calls with fix_low=false belong to the LOW path and are forwarded to the
// original builders unchanged. The view has a conversion back to
// std::vector<FBlock> so older LOW helper code with an explicit vector type keeps
// exactly its previous behavior.
class MaskShardHighFBlockView {
    const std::vector<FBlock>* cached_vec_ = nullptr;
    const FBlock* cached_raw_ = nullptr;
    std::size_t cached_raw_size_ = 0;
    std::vector<FBlock> owned_;

public:
    MaskShardHighFBlockView() = default;
    explicit MaskShardHighFBlockView(const std::vector<FBlock>* cached)
        : cached_vec_(cached) {}
    MaskShardHighFBlockView(const FBlock* cached, std::size_t n)
        : cached_raw_(cached), cached_raw_size_(n) {}
    explicit MaskShardHighFBlockView(std::vector<FBlock>&& owned)
        : owned_(std::move(owned)) {}

    std::size_t size() const {
        if (cached_raw_) return cached_raw_size_;
        if (cached_vec_) return cached_vec_->size();
        return owned_.size();
    }
    bool empty() const { return size() == 0; }
    const FBlock* data() const {
        if (cached_raw_) return cached_raw_;
        if (cached_vec_) return cached_vec_->data();
        return owned_.data();
    }
    const FBlock& back() const { return data()[size() - 1]; }
    const FBlock& operator[](std::size_t i) const { return data()[i]; }
    const FBlock* begin() const { return data(); }
    const FBlock* end() const { return data() + size(); }

    operator std::vector<FBlock>() && {
        if (!cached_vec_ && !cached_raw_) return std::move(owned_);
        return std::vector<FBlock>(begin(), end());
    }
    operator std::vector<FBlock>() const & {
        return std::vector<FBlock>(begin(), end());
    }
};

#ifdef MASKSHARD_HIGH_PINNED_CONFIG

// v0.62: replace many small std::vector payload allocations with two pinned,
// contiguous process-lifetime arenas. This makes the v0.61
// cudaMemcpyToSymbolAsync() FBlock sources eligible for true asynchronous DMA.
static constexpr std::size_t MS_HIGH_FBLOCK_MASKS = 1u << LOW_LUT_K;
static constexpr std::size_t MS_HIGH_MAIN_BLOCKS_PER_MASK =
    std::size_t(3) * std::size_t(HIGH_LUT_K + 2);
static constexpr std::size_t MS_HIGH_BLOCK_BLOCKS_PER_MASK =
    std::size_t(HIGH_LUT_K + 2);

#ifdef MASKSHARD_HIGH_FBLOCK_CLASS_CACHE
// v0.65: with every LOW position fixed, an unoccupied position is a forced N
// (identity) step and an occupied position is a +/- height step. Identity
// commutes with the height transition, so all per-height LOW counts -- and thus
// every HIGH FBlock off/end/stride -- depend only on popcount(mask). There are
// LOW_LUT_K+1 layouts rather than 2^LOW_LUT_K layouts.
static constexpr std::size_t MS_HIGH_FBLOCK_CACHE_SLOTS = LOW_LUT_K + 1;
static std::size_t maskshard_high_fblock_cache_slot(std::uint32_t mask) {
    std::size_t n = 0;
    while (mask) {
        mask &= mask - 1;
        ++n;
    }
    return n;
}
#else
static constexpr std::size_t MS_HIGH_FBLOCK_CACHE_SLOTS = MS_HIGH_FBLOCK_MASKS;
static std::size_t maskshard_high_fblock_cache_slot(std::uint32_t mask) {
    return std::size_t(mask);
}
#endif

static FBlock* G_MS_HIGH_FBLOCK_PINNED_MAIN = nullptr;
static FBlock* G_MS_HIGH_FBLOCK_PINNED_BLOCK = nullptr;
static std::array<std::uint8_t, MS_HIGH_FBLOCK_CACHE_SLOTS>
    G_MS_HIGH_FBLOCK_PINNED_MAIN_BUILT{};
static std::array<std::uint8_t, MS_HIGH_FBLOCK_CACHE_SLOTS>
    G_MS_HIGH_FBLOCK_PINNED_BLOCK_BUILT{};

static void maskshard_high_fblock_ensure_pinned_storage() {
    if (G_MS_HIGH_FBLOCK_PINNED_MAIN) return;
    void* mainp = nullptr;
    void* blockp = nullptr;
    const std::size_t main_bytes = MS_HIGH_FBLOCK_CACHE_SLOTS
        * MS_HIGH_MAIN_BLOCKS_PER_MASK * sizeof(FBlock);
    const std::size_t block_bytes = MS_HIGH_FBLOCK_CACHE_SLOTS
        * MS_HIGH_BLOCK_BLOCKS_PER_MASK * sizeof(FBlock);
    ck(cudaHostAlloc(&mainp, main_bytes, cudaHostAllocPortable),
       "HIGH FBlock pinned MAIN cache");
    ck(cudaHostAlloc(&blockp, block_bytes, cudaHostAllocPortable),
       "HIGH FBlock pinned BLOCKED cache");
    G_MS_HIGH_FBLOCK_PINNED_MAIN = static_cast<FBlock*>(mainp);
    G_MS_HIGH_FBLOCK_PINNED_BLOCK = static_cast<FBlock*>(blockp);
    std::cerr << "HIGH FBlock pinned cache allocated slots="
              << MS_HIGH_FBLOCK_CACHE_SLOTS
              << " main_mib="
              << double(main_bytes) / double(1ULL << 20)
              << " blocked_mib="
              << double(block_bytes) / double(1ULL << 20)
              << " total_mib="
              << double(main_bytes + block_bytes) / double(1ULL << 20)
              << '\n';
}

static MaskShardHighFBlockView maskshard_high_cached_main_blocks(
    bool fix_low, std::uint32_t mask
) {
    if (!fix_low)
        return MaskShardHighFBlockView(
            (make_factor_main_blocks)(false, mask));
    if (mask >= MS_HIGH_FBLOCK_MASKS) {
        std::cerr << "HIGH FBlock MAIN cache mask overflow mask=" << mask << '\n';
        std::exit(360);
    }
    maskshard_high_fblock_ensure_pinned_storage();
    const std::size_t slot = maskshard_high_fblock_cache_slot(mask);
    FBlock* const dst = G_MS_HIGH_FBLOCK_PINNED_MAIN
        + slot * MS_HIGH_MAIN_BLOCKS_PER_MASK;
    if (!G_MS_HIGH_FBLOCK_PINNED_MAIN_BUILT[slot]) {
        if (std::this_thread::get_id() != G_MS_HIGH_GROUP_SYNC_MAIN_THREAD) {
            std::cerr << "HIGH FBlock MAIN cache miss on worker mask=" << mask
                      << " slot=" << slot << '\n';
            std::exit(361);
        }
        const std::vector<FBlock> v = (make_factor_main_blocks)(true, mask);
        if (v.size() != MS_HIGH_MAIN_BLOCKS_PER_MASK) {
            std::cerr << "HIGH FBlock pinned MAIN size mismatch mask=" << mask
                      << " got=" << v.size()
                      << " expected=" << MS_HIGH_MAIN_BLOCKS_PER_MASK << '\n';
            std::exit(368);
        }
        std::copy(v.begin(), v.end(), dst);
        G_MS_HIGH_FBLOCK_PINNED_MAIN_BUILT[slot] = 1;
    }
    return MaskShardHighFBlockView(dst, MS_HIGH_MAIN_BLOCKS_PER_MASK);
}

static MaskShardHighFBlockView maskshard_high_cached_block_blocks(
    bool fix_low, std::uint32_t mask
) {
    if (!fix_low)
        return MaskShardHighFBlockView(
            (make_factor_block_blocks)(false, mask));
    if (mask >= MS_HIGH_FBLOCK_MASKS) {
        std::cerr << "HIGH FBlock BLOCKED cache mask overflow mask=" << mask << '\n';
        std::exit(362);
    }
    maskshard_high_fblock_ensure_pinned_storage();
    const std::size_t slot = maskshard_high_fblock_cache_slot(mask);
    FBlock* const dst = G_MS_HIGH_FBLOCK_PINNED_BLOCK
        + slot * MS_HIGH_BLOCK_BLOCKS_PER_MASK;
    if (!G_MS_HIGH_FBLOCK_PINNED_BLOCK_BUILT[slot]) {
        if (std::this_thread::get_id() != G_MS_HIGH_GROUP_SYNC_MAIN_THREAD) {
            std::cerr << "HIGH FBlock BLOCKED cache miss on worker mask=" << mask
                      << " slot=" << slot << '\n';
            std::exit(363);
        }
        const std::vector<FBlock> v = (make_factor_block_blocks)(true, mask);
        if (v.size() != MS_HIGH_BLOCK_BLOCKS_PER_MASK) {
            std::cerr << "HIGH FBlock pinned BLOCKED size mismatch mask=" << mask
                      << " got=" << v.size()
                      << " expected=" << MS_HIGH_BLOCK_BLOCKS_PER_MASK << '\n';
            std::exit(369);
        }
        std::copy(v.begin(), v.end(), dst);
        G_MS_HIGH_FBLOCK_PINNED_BLOCK_BUILT[slot] = 1;
        if (mask + 1 == MS_HIGH_FBLOCK_MASKS) {
            const std::size_t payload = MS_HIGH_FBLOCK_CACHE_SLOTS
                * (MS_HIGH_MAIN_BLOCKS_PER_MASK + MS_HIGH_BLOCK_BLOCKS_PER_MASK)
                * sizeof(FBlock);
            std::cerr << "HIGH FBlock cache masks=" << MS_HIGH_FBLOCK_MASKS
                      << " slots=" << MS_HIGH_FBLOCK_CACHE_SLOTS
                      << " payload_mib="
                      << double(payload) / double(1ULL << 20)
                      << " worker_rebuilds=0 pinned=1 contiguous=1\n";
        }
    }
    return MaskShardHighFBlockView(dst, MS_HIGH_BLOCK_BLOCKS_PER_MASK);
}

#else

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
                      << " worker_rebuilds=0 pinned=0\n";
        }
    }
    return MaskShardHighFBlockView(&e.block_blocks);
}

#endif

// Function-like macros do not expand the parenthesized original names used in
// the wrappers above.
#define make_factor_main_blocks(fix_low, mask) \
    maskshard_high_cached_main_blocks((fix_low), (mask))
#define make_factor_block_blocks(fix_low, mask) \
    maskshard_high_cached_block_blocks((fix_low), (mask))
