#pragma once

// Compatibility shim. The LOW center-partner fix is integrated into the base
// bucket kernel.  Keep this entry point for older probes; when the experimental
// pseudo-Mersenne accumulator is requested, route both LOW and subsequent HIGH
// selftest calls through the PM closure variants.

#if GPU_DIRECT_PM_ACCUM
#include "ramstream32_bucket_fused_pm.cuh"
#define bucket_run_high_fused bucket_run_high_fused_pm
static void bucket_run_low_fused_v2(
    const StorageLayout& layout,int threads=256,int grid_x=16,int grid_y=8
){
    bucket_run_low_fused_pm(layout,threads,grid_x,grid_y);
}
#else
static void bucket_run_low_fused_v2(
    const StorageLayout& layout,int threads=256,int grid_x=16,int grid_y=8
){
    bucket_run_low_fused(layout,threads,grid_x,grid_y);
}
#endif
