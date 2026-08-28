#pragma once

#include "ramstream32_bucket_orbit_closure_pattern10_depthcode_warpstriped_delta_direct_cross5.cuh"

static void p10dc_warpstriped_delta_direct_cross5_require_threads(int threads) {
    if (!p10dc_warpstriped_threads_ok(threads)) {
        std::cerr << "pattern10 depthcode warpstriped-delta-direct-cross5 requires BUCKET_THREADS to be a multiple of 32 in [32,1024], got "
                  << threads << '\n';
        std::exit(613);
    }
}

static void bucket_enqueue_low_orbit_closure_pattern10_depthcode_warpstriped_delta_direct_cross5(
    const StorageLayout& layout, cudaStream_t stream,
    int threads = 256, int gx = 16, int gy = 8
) {
    dim3 block(threads), grid(gx, gy, unsigned(layout.main_blocks.size()));
    for (int p = LOW_LUT_K; p >= 1; --p) {
        bucket_low_orbit_closure_pattern10_depthcode_kernel<<<grid, block, 0, stream>>>(p);
        ck(cudaGetLastError(), "bucket low pattern10 depthcode warpstriped-delta-direct-cross5 stream");
    }
}

static void bucket_enqueue_high_orbit_closure_pattern10_depthcode_warpstriped_delta_direct_cross5(
    const StorageLayout& layout, cudaStream_t stream,
    int threads = 256, int gx = 16, int gy = 8
) {
    p10dc_warpstriped_delta_direct_cross5_require_threads(threads);
    dim3 block(threads), grid(gx, gy, unsigned(layout.main_blocks.size()));
    const size_t smem = p10dc_warpctx_smem_bytes(threads);
    for (int p = TARGET_W - 1; p >= LOW_LUT_K + 1; --p) {
        bucket_high_orbit_closure_pattern10_depthcode_warpstriped_delta_direct_cross5_kernel<<<grid, block, smem, stream>>>(p);
        ck(cudaGetLastError(), "bucket high pattern10 depthcode warpstriped-delta-direct-cross5 stream");
    }
}

static void bucket_enqueue_reverse_low_pattern10_depthcode_warpstriped_delta_direct_cross5(
    const StorageLayout& layout, cudaStream_t stream,
    int threads = 256, int gx = 16, int gy = 8
) {
    dim3 block(threads), grid(gx, gy, unsigned(layout.main_blocks.size()));
    for (int p = 1; p <= LOW_LUT_K; ++p) {
        bucket_reverse_low_pattern10_depthcode_kernel<<<grid, block, 0, stream>>>(p);
        ck(cudaGetLastError(), "bucket reverse low pattern10 depthcode warpstriped-delta-direct-cross5 stream");
    }
}

static void bucket_enqueue_reverse_high_pattern10_depthcode_warpstriped_delta_direct_cross5(
    const StorageLayout& layout, cudaStream_t stream,
    int threads = 256, int gx = 16, int gy = 8
) {
    p10dc_warpstriped_delta_direct_cross5_require_threads(threads);
    dim3 block(threads), grid(gx, gy, unsigned(layout.main_blocks.size()));
    const size_t smem = p10dc_warpctx_smem_bytes(threads);
    for (int p = LOW_LUT_K + 1; p < TARGET_W; ++p) {
        bucket_reverse_high_pattern10_depthcode_warpstriped_delta_direct_cross5_kernel<<<grid, block, smem, stream>>>(p);
        ck(cudaGetLastError(), "bucket reverse high pattern10 depthcode warpstriped-delta-direct-cross5 stream");
    }
}
