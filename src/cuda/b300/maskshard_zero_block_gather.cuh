#pragma once

#ifndef MASKSHARD_SKIP_ZERO_BLOCK_GATHER
#error "maskshard_zero_block_gather.cuh requires MASKSHARD_SKIP_ZERO_BLOCK_GATHER"
#endif

// At a DP row boundary the authoritative BLOCKED vector is identically zero:
// the previous row ended at p=1, where no MAIN included transition can produce
// BLOCKED output and every pre-existing BLOCKED state is excluded back to MAIN.
//
// The normal HIGH gather nevertheless reads every authoritative BLOCKED state
// through P2P. v0.12 replaces that read with local zero stores. If
// MASKSHARD_LAZY_ZERO_BLOCK_INIT is also enabled, the first HIGH orbit treats
// the old BLOCKED value as zero and overwrites every BLOCKED coordinate before
// closure, so even the local zero stores can be skipped.
template<bool SCATTER>
__global__ void maskshard_high_block_io_skipzero_kernel(Count* scratch, Code n) {
#ifdef MASKSHARD_LAZY_ZERO_BLOCK_INIT
    if constexpr (!SCATTER) {
        (void)scratch;
        (void)n;
        return;
    }
#endif
    Code i = Code(blockIdx.x) * blockDim.x + threadIdx.x;
    const Code step = Code(gridDim.x) * blockDim.x;
    for (; i < n; i += step) {
        if constexpr (SCATTER) {
            const int bid = f_find_block(i);
            const FBlock x = D_F_BLOCK_BLOCKS[bid];
            uint32_t hr = 0, lr = 0;
            maskshard_split_rank(i, x, hr, lr);
            Count* p = maskshard_block_addr(bid, hr, lr);
            *p = scratch[i];
        } else {
            scratch[i] = 0;
        }
    }
}

#define maskshard_high_block_io_kernel maskshard_high_block_io_skipzero_kernel
