#pragma once

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

// The mask-shard HIGH kernels use FBlocks, D_F_MASK and precomputed storage/
// descriptor tables. They do not call generic factor_rank/unrank helpers, so
// the seven legacy GroupSpec symbols above are dead on this path. Restrict the
// elision to non-main HIGH worker threads; setup/reset and any future main-thread
// use of the same symbols still goes through the real CUDA runtime call.
template<class Symbol>
static cudaError_t maskshard_high_filtered_memcpy_to_symbol(
    const char* name,
    const Symbol& symbol,
    const void* src,
    std::size_t count,
    std::size_t offset = 0,
    cudaMemcpyKind kind = cudaMemcpyHostToDevice
) {
    if (std::this_thread::get_id() != G_MS_HIGH_GROUP_SYNC_MAIN_THREAD
        && maskshard_high_symbol_is_dead(name))
        return cudaSuccess;
    return cudaMemcpyToSymbol(symbol, src, count, offset, kind);
}

// Keep the real cudaMemcpyToSymbol() inside the wrapper above unexpanded.
#define cudaMemcpyToSymbol(symbol, src, count, ...) \
    maskshard_high_filtered_memcpy_to_symbol( \
        #symbol, symbol, src, count, ##__VA_ARGS__)
