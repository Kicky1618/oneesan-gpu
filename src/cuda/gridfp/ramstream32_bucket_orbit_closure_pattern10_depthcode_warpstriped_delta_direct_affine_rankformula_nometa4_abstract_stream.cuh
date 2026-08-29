#pragma once

#include "ramstream32_bucket_orbit_closure_pattern10_depthcode_warpstriped_delta_direct_affine_rankformula_nometa4_abstract.cuh"

static void p10dc_warpstriped_delta_direct_affine_rankformula_nometa4_abstract_require_threads(int threads) {
    if (!p10dc_warpstriped_threads_ok(threads)) {
        std::cerr << "pattern10 depthcode rankformula-nometa4-abstract requires BUCKET_THREADS multiple of 32 in [32,1024], got " << threads << '\n';
        std::exit(774);
    }
}

static void p10dc_rankformula_nometa4_abstract_configure_high_smem(int threads) {
    const size_t smem = p10dc_direct_warpctx_smem_bytes(threads);
    int device = 0;
    ck(cudaGetDevice(&device), "rankformula smem get device");
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, device), "rankformula smem device props");
    const size_t default_limit = size_t(prop.sharedMemPerBlock);
    const size_t optin_limit = size_t(prop.sharedMemPerBlockOptin);
    if (smem > optin_limit) {
        std::cerr << "rankformula HIGH dynamic shared memory exceeds device opt-in limit"
                  << " device=" << device
                  << " threads=" << threads
                  << " requested=" << smem
                  << " default_limit=" << default_limit
                  << " optin_limit=" << optin_limit << '\n';
        std::exit(773);
    }
    if (smem > default_limit) {
        ck(cudaFuncSetAttribute(
               bucket_high_orbit_closure_pattern10_depthcode_warpstriped_delta_direct_affine_rankformula_nometa4_abstract_kernel,
               cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem)),
           "rankformula forward HIGH opt-in smem");
        ck(cudaFuncSetAttribute(
               bucket_reverse_high_pattern10_depthcode_warpstriped_delta_direct_affine_rankformula_nometa4_abstract_kernel,
               cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem)),
           "rankformula reverse HIGH opt-in smem");
        std::cerr << "rankformula_high_smem_optin device=" << device
                  << " threads=" << threads
                  << " dynamic_smem_bytes=" << smem
                  << " default_limit=" << default_limit
                  << " optin_limit=" << optin_limit << '\n';
    }
}

static void p10dc_rankformula_nometa4_abstract_report_high_occupancy(int threads) {
    p10dc_rankformula_nometa4_abstract_configure_high_smem(threads);
    const size_t smem = p10dc_direct_warpctx_smem_bytes(threads);
    int device = 0;
    ck(cudaGetDevice(&device), "rankformula occupancy get device");
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, device), "rankformula occupancy device props");
    int fblocks = 0, rblocks = 0;
    ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
           &fblocks,
           bucket_high_orbit_closure_pattern10_depthcode_warpstriped_delta_direct_affine_rankformula_nometa4_abstract_kernel,
           threads, smem),
       "rankformula forward HIGH occupancy");
    ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
           &rblocks,
           bucket_reverse_high_pattern10_depthcode_warpstriped_delta_direct_affine_rankformula_nometa4_abstract_kernel,
           threads, smem),
       "rankformula reverse HIGH occupancy");
    cudaFuncAttributes fa{}, ra{};
    ck(cudaFuncGetAttributes(
           &fa,
           bucket_high_orbit_closure_pattern10_depthcode_warpstriped_delta_direct_affine_rankformula_nometa4_abstract_kernel),
       "rankformula forward HIGH attributes");
    ck(cudaFuncGetAttributes(
           &ra,
           bucket_reverse_high_pattern10_depthcode_warpstriped_delta_direct_affine_rankformula_nometa4_abstract_kernel),
       "rankformula reverse HIGH attributes");
    const int warp_cap = prop.maxThreadsPerMultiProcessor / prop.warpSize;
    const int fwarps = fblocks * (threads / prop.warpSize);
    const int rwarps = rblocks * (threads / prop.warpSize);
    std::cerr << "rankformula_high_occupancy device=" << device
              << " threads=" << threads
              << " dynamic_smem_bytes=" << smem
              << " forward_regs=" << fa.numRegs
              << " reverse_regs=" << ra.numRegs
              << " forward_static_smem=" << fa.sharedSizeBytes
              << " reverse_static_smem=" << ra.sharedSizeBytes
              << " forward_blocks_per_sm=" << fblocks
              << " reverse_blocks_per_sm=" << rblocks
              << " forward_warps_per_sm=" << fwarps
              << " reverse_warps_per_sm=" << rwarps
              << " warp_cap=" << warp_cap
              << " forward_warp_occupancy_pct=" << (warp_cap ? 100.0 * double(fwarps) / double(warp_cap) : 0.0)
              << " reverse_warp_occupancy_pct=" << (warp_cap ? 100.0 * double(rwarps) / double(warp_cap) : 0.0)
              << '\n';
}

static void bucket_enqueue_low_orbit_closure_pattern10_depthcode_warpstriped_delta_direct_affine_rankformula_nometa4_abstract(const StorageLayout& layout,cudaStream_t stream,int threads=256,int gx=16,int gy=8){dim3 block(threads),grid(gx,gy,unsigned(layout.main_blocks.size()));for(int p=LOW_LUT_K;p>=1;--p){bucket_low_orbit_closure_pattern10_depthcode_kernel<<<grid,block,0,stream>>>(p);ck(cudaGetLastError(),"bucket low rankformula-nometa4-abstract stream");}}
static void bucket_enqueue_high_orbit_closure_pattern10_depthcode_warpstriped_delta_direct_affine_rankformula_nometa4_abstract(const StorageLayout& layout,cudaStream_t stream,int threads=256,int gx=16,int gy=8){p10dc_warpstriped_delta_direct_affine_rankformula_nometa4_abstract_require_threads(threads);p10dc_rankformula_nometa4_abstract_configure_high_smem(threads);dim3 block(threads),grid(gx,gy,unsigned(layout.main_blocks.size()));const size_t smem=p10dc_direct_warpctx_smem_bytes(threads);for(int p=TARGET_W-1;p>=LOW_LUT_K+1;--p){bucket_high_orbit_closure_pattern10_depthcode_warpstriped_delta_direct_affine_rankformula_nometa4_abstract_kernel<<<grid,block,smem,stream>>>(p);ck(cudaGetLastError(),"bucket high rankformula-nometa4-abstract stream");}}
static void bucket_enqueue_reverse_low_pattern10_depthcode_warpstriped_delta_direct_affine_rankformula_nometa4_abstract(const StorageLayout& layout,cudaStream_t stream,int threads=256,int gx=16,int gy=8){dim3 block(threads),grid(gx,gy,unsigned(layout.main_blocks.size()));for(int p=1;p<=LOW_LUT_K;++p){bucket_reverse_low_pattern10_depthcode_kernel<<<grid,block,0,stream>>>(p);ck(cudaGetLastError(),"bucket reverse low rankformula-nometa4-abstract stream");}}
static void bucket_enqueue_reverse_high_pattern10_depthcode_warpstriped_delta_direct_affine_rankformula_nometa4_abstract(const StorageLayout& layout,cudaStream_t stream,int threads=256,int gx=16,int gy=8){p10dc_warpstriped_delta_direct_affine_rankformula_nometa4_abstract_require_threads(threads);p10dc_rankformula_nometa4_abstract_configure_high_smem(threads);dim3 block(threads),grid(gx,gy,unsigned(layout.main_blocks.size()));const size_t smem=p10dc_direct_warpctx_smem_bytes(threads);for(int p=LOW_LUT_K+1;p<TARGET_W;++p){bucket_reverse_high_pattern10_depthcode_warpstriped_delta_direct_affine_rankformula_nometa4_abstract_kernel<<<grid,block,smem,stream>>>(p);ck(cudaGetLastError(),"bucket reverse high rankformula-nometa4-abstract stream");}}
