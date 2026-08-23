#pragma once

#include <array>
#include <cstdint>
#include <iostream>
#include <vector>
#include "maskshard_index.cuh"

// Direct HIGH-window I/O for HIGH-mask-sharded authoritative HBM.
//
// Scratch layout (fix_low=true):
//   row = HIGH all-rank, col = LOW rank within the fixed LOW occupancy mask.
// Authoritative layout:
//   owner = owner[HIGH occupancy mask]
//   row = HIGH rank within that occupancy mask
//   col = LOW storage all-rank
//
// Therefore the conversion is purely factorized. No MateID and no canonical
// full-frontier rank are constructed.

__constant__ Count* D_MS_MAIN_PTR[8];
__constant__ Count* D_MS_BLOCK_PTR[8];
__constant__ const uint8_t* D_MS_OWNER;
__constant__ const Code* D_MS_MAIN_BASE;
__constant__ const Code* D_MS_BLOCK_BASE;
__constant__ const Code* D_MS_MAIN_BLOCK_OFF;
__constant__ const Code* D_MS_BLOCK_BLOCK_OFF;
__constant__ const uint32_t* D_MS_HIGH_ROUTE;
__constant__ const uint32_t* D_MS_LOW_BEGIN;
__constant__ uint32_t D_MS_MAIN_NBLOCKS;
__constant__ uint32_t D_MS_BLOCK_NBLOCKS;
__constant__ uint32_t D_MS_MAIN_COLS[64];
__constant__ uint32_t D_MS_BLOCK_COLS[32];

struct MaskShardDeviceMeta {
    int dev = -1;
    uint8_t* owner = nullptr;
    Code* main_base = nullptr;
    Code* block_base = nullptr;
    Code* main_block_off = nullptr;
    Code* block_block_off = nullptr;
    uint32_t* high_route = nullptr;
    uint32_t* low_begin = nullptr;

    template<class T>
    static void copy_vec(T** dst, const std::vector<T>& v, const char* what) {
        if (v.empty()) return;
        ck(cudaMalloc(dst, v.size() * sizeof(T)), what);
        ck(cudaMemcpy(*dst, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), what);
    }

    void install(
        int device,
        const MaskShardLayout& shard,
        const StorageLayout& layout,
        const std::vector<uint32_t>& route,
        Count* const main_ptr[8],
        Count* const block_ptr[8]
    ) {
        dev = device;
        ck(cudaSetDevice(dev), "maskshard meta set device");
        copy_vec(&owner, shard.owner, "maskshard owner");
        copy_vec(&main_base, shard.main_base, "maskshard main base");
        copy_vec(&block_base, shard.block_base, "maskshard block base");
        copy_vec(&main_block_off, shard.main_block_off, "maskshard main block off");
        copy_vec(&block_block_off, shard.block_block_off, "maskshard block block off");
        copy_vec(&high_route, route, "maskshard high route");

        constexpr int S = MAXW + 2;
        constexpr uint32_t NM = 1u << LOW_LUT_K;
        std::vector<uint32_t> lb(size_t(NM) * S, 0u);
        for (int h = 0; h <= LOW_LUT_K + 1; ++h) {
            uint32_t rank = 0;
            for (uint32_t mask = 0; mask < NM; ++mask) {
                const size_t ix = size_t(mask) * S + h;
                lb[ix] = rank;
                rank += G_FACTOR.low_mask_off[ix + 1] - G_FACTOR.low_mask_off[ix];
            }
            const uint32_t expected = G_FACTOR.low_all_off[h + 1] - G_FACTOR.low_all_off[h];
            if (rank != expected) {
                std::cerr << "maskshard LOW storage begin mismatch h=" << h
                          << " got=" << rank << " expected=" << expected << '\n';
                std::exit(129);
            }
        }
        copy_vec(&low_begin, lb, "maskshard low storage begin");

        ck(cudaMemcpyToSymbol(D_MS_MAIN_PTR, main_ptr, sizeof(Count*) * 8), "maskshard main ptrs");
        ck(cudaMemcpyToSymbol(D_MS_BLOCK_PTR, block_ptr, sizeof(Count*) * 8), "maskshard block ptrs");
        ck(cudaMemcpyToSymbol(D_MS_OWNER, &owner, sizeof(owner)), "maskshard owner ptr");
        ck(cudaMemcpyToSymbol(D_MS_MAIN_BASE, &main_base, sizeof(main_base)), "maskshard main base ptr");
        ck(cudaMemcpyToSymbol(D_MS_BLOCK_BASE, &block_base, sizeof(block_base)), "maskshard block base ptr");
        ck(cudaMemcpyToSymbol(D_MS_MAIN_BLOCK_OFF, &main_block_off, sizeof(main_block_off)), "maskshard main block off ptr");
        ck(cudaMemcpyToSymbol(D_MS_BLOCK_BLOCK_OFF, &block_block_off, sizeof(block_block_off)), "maskshard block block off ptr");
        ck(cudaMemcpyToSymbol(D_MS_HIGH_ROUTE, &high_route, sizeof(high_route)), "maskshard high route ptr");
        ck(cudaMemcpyToSymbol(D_MS_LOW_BEGIN, &low_begin, sizeof(low_begin)), "maskshard low begin ptr");
        ck(cudaMemcpyToSymbol(D_MS_MAIN_NBLOCKS, &shard.main_nblocks, sizeof(shard.main_nblocks)), "maskshard main nblocks");
        ck(cudaMemcpyToSymbol(D_MS_BLOCK_NBLOCKS, &shard.block_nblocks, sizeof(shard.block_nblocks)), "maskshard block nblocks");

        std::array<uint32_t, 64> mc{};
        std::array<uint32_t, 32> bc{};
        for (size_t i = 0; i < layout.main_blocks.size(); ++i) mc[i] = layout.main_blocks[i].cols;
        for (size_t i = 0; i < layout.block_blocks.size(); ++i) bc[i] = layout.block_blocks[i].cols;
        ck(cudaMemcpyToSymbol(D_MS_MAIN_COLS, mc.data(), sizeof(mc)), "maskshard main cols");
        ck(cudaMemcpyToSymbol(D_MS_BLOCK_COLS, bc.data(), sizeof(bc)), "maskshard block cols");
    }

    void release() {
        if (dev < 0) return;
        cudaSetDevice(dev);
        if (owner) cudaFree(owner);
        if (main_base) cudaFree(main_base);
        if (block_base) cudaFree(block_base);
        if (main_block_off) cudaFree(main_block_off);
        if (block_block_off) cudaFree(block_block_off);
        if (high_route) cudaFree(high_route);
        if (low_begin) cudaFree(low_begin);
        owner = nullptr;
        main_base = block_base = nullptr;
        main_block_off = block_block_off = nullptr;
        high_route = low_begin = nullptr;
        dev = -1;
    }
};

__device__ __forceinline__ uint32_t maskshard_low_all_rank(
    uint32_t low_mask, uint32_t hs, uint32_t low_mask_rank
) {
    constexpr int S = MAXW + 2;
    return D_MS_LOW_BEGIN[size_t(low_mask) * S + hs] + low_mask_rank;
}

__device__ __forceinline__ void maskshard_high_route(
    uint32_t he, uint32_t high_all_rank, uint32_t& mask, uint32_t& mask_rank
) {
    constexpr uint32_t HM = (1u << HIGH_LUT_K) - 1u;
    const uint32_t packed = D_MS_HIGH_ROUTE[D_F_HIGH_ALL_OFF[he] + high_all_rank];
    mask = packed & HM;
    mask_rank = packed >> HIGH_LUT_K;
}

__device__ __forceinline__ Count* maskshard_main_addr(
    int bid, uint32_t high_all_rank, uint32_t low_mask_rank
) {
    const FBlock x = D_F_MAIN_BLOCKS[bid];
    uint32_t mask = 0, mr = 0;
    maskshard_high_route(x.he, high_all_rank, mask, mr);
    const uint32_t lar = maskshard_low_all_rank(D_F_MASK, x.hs, low_mask_rank);
    const int owner = D_MS_OWNER[mask];
    const Code off = D_MS_MAIN_BASE[mask]
        + D_MS_MAIN_BLOCK_OFF[size_t(mask) * D_MS_MAIN_NBLOCKS + bid]
        + Code(mr) * D_MS_MAIN_COLS[bid] + lar;
    return D_MS_MAIN_PTR[owner] + off;
}

__device__ __forceinline__ Count* maskshard_block_addr(
    int bid, uint32_t high_all_rank, uint32_t low_mask_rank
) {
    const FBlock x = D_F_BLOCK_BLOCKS[bid];
    uint32_t mask = 0, mr = 0;
    maskshard_high_route(x.he, high_all_rank, mask, mr);
    const uint32_t lar = maskshard_low_all_rank(D_F_MASK, x.hs, low_mask_rank);
    const int owner = D_MS_OWNER[mask];
    const Code off = D_MS_BLOCK_BASE[mask]
        + D_MS_BLOCK_BLOCK_OFF[size_t(mask) * D_MS_BLOCK_NBLOCKS + bid]
        + Code(mr) * D_MS_BLOCK_COLS[bid] + lar;
    return D_MS_BLOCK_PTR[owner] + off;
}

template<bool SCATTER>
__global__ void maskshard_high_main_io_kernel(Count* scratch, Code n) {
    Code i = Code(blockIdx.x) * blockDim.x + threadIdx.x;
    const Code step = Code(gridDim.x) * blockDim.x;
    for (; i < n; i += step) {
        const int bid = f_find_main(i);
        const FBlock x = D_F_MAIN_BLOCKS[bid];
        uint32_t hr = 0, lr = 0;
        maskshard_split_rank(i, x, hr, lr);
        Count* p = maskshard_main_addr(bid, hr, lr);
        if constexpr (SCATTER) *p = scratch[i];
        else scratch[i] = *p;
    }
}

template<bool SCATTER>
__global__ void maskshard_high_block_io_kernel(Count* scratch, Code n) {
    Code i = Code(blockIdx.x) * blockDim.x + threadIdx.x;
    const Code step = Code(gridDim.x) * blockDim.x;
    for (; i < n; i += step) {
        const int bid = f_find_block(i);
        const FBlock x = D_F_BLOCK_BLOCKS[bid];
        uint32_t hr = 0, lr = 0;
        maskshard_split_rank(i, x, hr, lr);
        Count* p = maskshard_block_addr(bid, hr, lr);
        if constexpr (SCATTER) *p = scratch[i];
        else scratch[i] = *p;
    }
}
