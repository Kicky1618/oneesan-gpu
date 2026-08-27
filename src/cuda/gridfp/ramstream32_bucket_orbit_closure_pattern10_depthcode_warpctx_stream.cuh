#pragma once

#include "ramstream32_bucket_orbit_closure_pattern10_depthcode_warpctx.cuh"

static void bucket_enqueue_low_orbit_closure_pattern10_depthcode_warpctx(
    const StorageLayout& layout, cudaStream_t stream,
    int threads = 256, int gx = 16, int gy = 8
) {
    dim3 block(threads), grid(gx, gy, unsigned(layout.main_blocks.size()));
    for (int p = LOW_LUT_K; p >= 1; --p) {
        bucket_low_orbit_closure_pattern10_depthcode_kernel<<<grid, block, 0, stream>>>(p);
        ck(cudaGetLastError(), "bucket low pattern10 depthcode warpctx stream");
    }
}

static void bucket_enqueue_high_orbit_closure_pattern10_depthcode_warpctx(
    const StorageLayout& layout, cudaStream_t stream,
    int threads = 256, int gx = 16, int gy = 8
) {
    dim3 block(threads), grid(gx, gy, unsigned(layout.main_blocks.size()));
    for (int p = TARGET_W - 1; p >= LOW_LUT_K + 1; --p) {
        bucket_high_orbit_closure_pattern10_depthcode_warpctx_kernel<<<grid, block, 0, stream>>>(p);
        ck(cudaGetLastError(), "bucket high pattern10 depthcode warpctx stream");
    }
}

static void bucket_enqueue_reverse_low_pattern10_depthcode_warpctx(
    const StorageLayout& layout, cudaStream_t stream,
    int threads = 256, int gx = 16, int gy = 8
) {
    dim3 block(threads), grid(gx, gy, unsigned(layout.main_blocks.size()));
    for (int p = 1; p <= LOW_LUT_K; ++p) {
        bucket_reverse_low_pattern10_depthcode_kernel<<<grid, block, 0, stream>>>(p);
        ck(cudaGetLastError(), "bucket reverse low pattern10 depthcode warpctx stream");
    }
}

static void bucket_enqueue_reverse_high_pattern10_depthcode_warpctx(
    const StorageLayout& layout, cudaStream_t stream,
    int threads = 256, int gx = 16, int gy = 8
) {
    dim3 block(threads), grid(gx, gy, unsigned(layout.main_blocks.size()));
    for (int p = LOW_LUT_K + 1; p < TARGET_W; ++p) {
        bucket_reverse_high_pattern10_depthcode_warpctx_kernel<<<grid, block, 0, stream>>>(p);
        ck(cudaGetLastError(), "bucket reverse high pattern10 depthcode warpctx stream");
    }
}
