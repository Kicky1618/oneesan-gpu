#pragma once

#include <array>
#include <cstddef>
#include <cstring>
#include <thread>

#ifndef MASKSHARD_HIGH_DEAD_SYMBOL_COPIES
#error "HIGH dead-symbol filter requires MASKSHARD_HIGH_DEAD_SYMBOL_COPIES"
#endif
#ifndef MASKSHARD_HIGH_GROUP_SYNC
#error "HIGH dead-symbol filter currently layers on v0.57 worker scoping"
#endif

static bool maskshard_high_symbol_is_dead(const char* name) {
    return std::strcmp(name, "D_F_FIX_LOW") == 0
        || std::strcmp(name, "D_MAIN_FIXED") == 0
        || std::strcmp(name, "D_MAIN_OCC") == 0
        || std::strcmp(name, "D_BLOCK_FIXED") == 0
        || std::strcmp(name, "D_BLOCK_OCC") == 0
        || std::strcmp(name, "D_MAIN_DP") == 0
        || std::strcmp(name, "D_BLOCK_DP") == 0;
}

#ifdef MASKSHARD_HIGH_ROW_BATCH_ASYNC
#ifndef MASKSHARD_HIGH_FBLOCK_CACHE
#error "async HIGH constant updates require persistent v0.59 FBlock storage"
#endif
static const std::array<std::uint32_t, (1u << LOW_LUT_K)>
    G_MS_HIGH_ASYNC_MASK_VALUE = [] {
        std::array<std::uint32_t, (1u << LOW_LUT_K)> a{};
        for (std::uint32_t i = 0; i < a.size(); ++i) a[i] = i;
        return a;
    }();
static const std::array<int, 65> G_MS_HIGH_ASYNC_SMALL_INT = [] {
    std::array<int, 65> a{};
    for (int i = 0; i < int(a.size()); ++i) a[std::size_t(i)] = i;
    return a;
}();
#endif

#ifdef MASKSHARD_HIGH_NBLOCK_ONCE
#ifndef MASKSHARD_HIGH_ROW_BATCH_ASYNC
#error "HIGH nblock-once optimization requires row-batch async worker lifetime"
#endif
#ifndef MASKSHARD_HIGH_PINNED_CONFIG
#error "HIGH nblock-once optimization requires v0.62 fixed pinned FBlock shapes"
#endif
static thread_local bool G_MS_HIGH_MAIN_NBLOCK_SENT = false;
static thread_local bool G_MS_HIGH_BLOCK_NBLOCK_SENT = false;
#endif

#ifdef MASKSHARD_HIGH_FBLOCK_LAYOUT_DEDUP
#ifndef MASKSHARD_HIGH_ROW_BATCH_ASYNC
#error "HIGH FBlock layout dedup requires row-batch async worker lifetime"
#endif
#ifndef MASKSHARD_HIGH_PINNED_CONFIG
#error "HIGH FBlock layout dedup requires persistent pinned FBlock sources"
#endif
static thread_local const void* G_MS_HIGH_LAST_MAIN_FBLOCKS = nullptr;
static thread_local const void* G_MS_HIGH_LAST_BLOCK_FBLOCKS = nullptr;

static bool maskshard_high_fblock_layout_redundant(
    const char* name,
    const void* src,
    std::size_t count,
    std::size_t offset,
    cudaMemcpyKind kind
) {
    if (offset != 0 || kind != cudaMemcpyHostToDevice) return false;

    const void** last = nullptr;
    std::size_t expected = 0;
    if (std::strcmp(name, "D_F_MAIN_BLOCKS") == 0) {
        last = &G_MS_HIGH_LAST_MAIN_FBLOCKS;
        expected = std::size_t(3 * (HIGH_LUT_K + 2)) * sizeof(FBlock);
    } else if (std::strcmp(name, "D_F_BLOCK_BLOCKS") == 0) {
        last = &G_MS_HIGH_LAST_BLOCK_FBLOCKS;
        expected = std::size_t(HIGH_LUT_K + 2) * sizeof(FBlock);
    } else {
        return false;
    }
    if (count != expected) return false;

    // v0.62 keeps every source array pinned and immutable through the whole
    // solve, so remembering its pointer is safe until this row-worker exits.
    // v0.65 aliases every mask of one occupancy-count class to the same pinned
    // slot, making pointer equality the common fast path. The bytewise fallback
    // preserves v0.64 behavior when per-mask storage is still in use.
    if (*last == src) return true;
    if (*last && std::memcmp(*last, src, count) == 0) return true;
    *last = src;
    return false;
}
#endif

// The mask-shard HIGH kernels use FBlocks, D_F_MASK and precomputed storage/
// descriptor tables. They do not call generic factor_rank/unrank helpers, so
// the seven legacy GroupSpec symbols above are dead on this path. Restrict the
// elision to non-main HIGH worker threads; setup/reset and any future main-thread
// use of the same symbols still goes through the real CUDA runtime call.
//
// v0.61 optionally queues the five retained HIGH config symbols asynchronously
// on the same default stream as the HIGH kernels. FBlock payloads are persistent
// in v0.59. The three scalar stack sources are redirected to immutable tables
// so their lifetime also extends until the row-end synchronization.
//
// v0.63 observes that v0.62 fixes the HIGH FBlock cardinalities for every mask:
// MAIN has 3*(HIGH_LUT_K+2) blocks and BLOCKED has HIGH_LUT_K+2. LOW work runs
// on separate threads after each HIGH row, so each fresh HIGH row-worker only
// needs to restore those two scalar symbols for its first job.
//
// v0.64 exploits a stronger invariant without changing any kernel: for fixed
// LOW_LUT_K the FBlock geometry depends only on the number of occupied LOW
// positions. build_fullorbit_batch_high_jobs() sorts by total work, which is
// monotone in that occupancy count for the n=27 target. A row-worker therefore
// sees long runs of byte-identical pinned FBlock arrays. Skip an async constant
// copy when its payload is identical to the last payload already queued on that
// worker's default stream. The comparison remains correct even if scheduling
// changes and layouts are not grouped; it merely loses some dedup opportunities.
template<class Symbol>
static cudaError_t maskshard_high_filtered_memcpy_to_symbol(
    const char* name,
    const Symbol& symbol,
    const void* src,
    std::size_t count,
    std::size_t offset = 0,
    cudaMemcpyKind kind = cudaMemcpyHostToDevice
) {
    const bool worker =
        std::this_thread::get_id() != G_MS_HIGH_GROUP_SYNC_MAIN_THREAD;
    if (worker && maskshard_high_symbol_is_dead(name)) return cudaSuccess;
#ifdef MASKSHARD_HIGH_ROW_BATCH_ASYNC
    if (worker) {
#ifdef MASKSHARD_HIGH_FBLOCK_LAYOUT_DEDUP
        if (maskshard_high_fblock_layout_redundant(
                name, src, count, offset, kind))
            return cudaSuccess;
#endif
        const void* stable_src = src;
        if (std::strcmp(name, "D_F_MASK") == 0) {
            const std::uint32_t v = *static_cast<const std::uint32_t*>(src);
            if (v >= G_MS_HIGH_ASYNC_MASK_VALUE.size()) return cudaErrorInvalidValue;
            stable_src = &G_MS_HIGH_ASYNC_MASK_VALUE[v];
        } else if (std::strcmp(name, "D_F_MAIN_NBLOCKS") == 0
                   || std::strcmp(name, "D_F_BLOCK_NBLOCKS") == 0) {
            const int v = *static_cast<const int*>(src);
            if (v < 0 || v >= int(G_MS_HIGH_ASYNC_SMALL_INT.size()))
                return cudaErrorInvalidValue;
#ifdef MASKSHARD_HIGH_NBLOCK_ONCE
            if (std::strcmp(name, "D_F_MAIN_NBLOCKS") == 0) {
                constexpr int expected = 3 * (HIGH_LUT_K + 2);
                if (v != expected) return cudaErrorInvalidValue;
                if (G_MS_HIGH_MAIN_NBLOCK_SENT) return cudaSuccess;
                G_MS_HIGH_MAIN_NBLOCK_SENT = true;
            } else {
                constexpr int expected = HIGH_LUT_K + 2;
                if (v != expected) return cudaErrorInvalidValue;
                if (G_MS_HIGH_BLOCK_NBLOCK_SENT) return cudaSuccess;
                G_MS_HIGH_BLOCK_NBLOCK_SENT = true;
            }
#endif
            stable_src = &G_MS_HIGH_ASYNC_SMALL_INT[std::size_t(v)];
        }
        return cudaMemcpyToSymbolAsync(
            symbol, stable_src, count, offset, kind, cudaStream_t(0));
    }
#endif
    return cudaMemcpyToSymbol(symbol, src, count, offset, kind);
}

// Keep the real cudaMemcpyToSymbol() inside the wrapper above unexpanded.
#define cudaMemcpyToSymbol(symbol, src, count, ...) \
    maskshard_high_filtered_memcpy_to_symbol( \
        #symbol, symbol, src, count, ##__VA_ARGS__)
