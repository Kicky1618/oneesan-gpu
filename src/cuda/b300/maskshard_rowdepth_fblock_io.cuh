#pragma once

#ifndef MASKSHARD_ROW_DEPTH_FBLOCK_IO
#error "maskshard_rowdepth_fblock_io.cuh requires MASKSHARD_ROW_DEPTH_FBLOCK_IO"
#endif

// Cheap structural-zero filter for early DP rows.  A factorized state in one
// FBlock shares HIGH ending height x.he and LOW starting height x.hs with all
// other states in the block.  If either exceeds the current row-depth cap, the
// complete state necessarily has max frontier height above the reachable cap
// and is therefore known zero.
//
// This is intentionally coarser than the exact max-height predicate: paths can
// rise above the cap and return while keeping he/hs small.  The benefit is that
// the test needs no new per-code metadata and only inspects the FBlock bytes.
__device__ __constant__ int D_MS_ROW_DEPTH_INDEX;

static void maskshard_set_row_depth_fblock_io_row(int zero_based_row) {
    ck(cudaMemcpyToSymbol(D_MS_ROW_DEPTH_INDEX, &zero_based_row,
                          sizeof(zero_based_row)),
       "maskshard row-depth row index");
}

__device__ __forceinline__ int maskshard_row_depth_io_cap(bool scatter) {
    const int row = D_MS_ROW_DEPTH_INDEX;
    return scatter ? row + 1 : (row > 0 ? row : 1);
}

template<bool SCATTER>
__global__ void maskshard_high_main_io_rowdepth_fblock_kernel(Count* scratch, Code n) {
    Code i = Code(blockIdx.x) * blockDim.x + threadIdx.x;
    const Code step = Code(gridDim.x) * blockDim.x;
    const int cap = maskshard_row_depth_io_cap(SCATTER);
    for (; i < n; i += step) {
        const int bid = f_find_main(i);
        const FBlock x = D_F_MAIN_BLOCKS[bid];
        const bool reachable_block = int(x.he) <= cap && int(x.hs) <= cap;
        if (!reachable_block) {
            if constexpr (!SCATTER) scratch[i] = 0;
            continue;
        }
        uint32_t hr = 0, lr = 0;
        maskshard_split_rank(i, x, hr, lr);
        Count* p = maskshard_main_addr(bid, hr, lr);
        if constexpr (SCATTER) *p = scratch[i];
        else scratch[i] = *p;
    }
}

template<bool SCATTER>
__global__ void maskshard_high_block_io_rowdepth_fblock_kernel(Count* scratch, Code n) {
    Code i = Code(blockIdx.x) * blockDim.x + threadIdx.x;
    const Code step = Code(gridDim.x) * blockDim.x;
    const int cap = maskshard_row_depth_io_cap(SCATTER);
    for (; i < n; i += step) {
        const int bid = f_find_block(i);
        const FBlock x = D_F_BLOCK_BLOCKS[bid];
        if (int(x.he) > cap) {
            if constexpr (!SCATTER) scratch[i] = 0;
            continue;
        }
        if constexpr (!SCATTER) {
            // v0.14 is layered on v0.13, so the shared host compiles the
            // gather-side BLOCKED launch out.  Keep this specialization safe if
            // called in isolation: row-boundary BLOCKED is known zero.
            scratch[i] = 0;
        } else {
            uint32_t hr = 0, lr = 0;
            maskshard_split_rank(i, x, hr, lr);
            Count* p = maskshard_block_addr(bid, hr, lr);
            *p = scratch[i];
        }
    }
}

#define maskshard_high_main_io_kernel \
        maskshard_high_main_io_rowdepth_fblock_kernel
#ifdef maskshard_high_block_io_kernel
#undef maskshard_high_block_io_kernel
#endif
#define maskshard_high_block_io_kernel \
        maskshard_high_block_io_rowdepth_fblock_kernel
