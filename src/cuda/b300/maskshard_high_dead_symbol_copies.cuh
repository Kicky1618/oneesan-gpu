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

// The mask-shard HIGH kernels use FBlocks, D_F_MASK and precomputed storage/
// descriptor tables. They do not call generic factor_rank/unrank helpers, so
// the seven legacy GroupSpec symbols above are dead on this path. Restrict the
// elision to non-main HIGH worker threads; setup/reset and any future main-thread
// use of the same symbols still goes through the real CUDA runtime call.
//
// v0.61 optionally queues the five retained HIGH config symbols asynchronously
// on the same default stream as the HIGH kernels.  FBlock payloads are persistent
// in v0.59.  The three scalar stack sources are redirected to immutable tables
// so their lifetime also extends until the row-end synchronization.
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
