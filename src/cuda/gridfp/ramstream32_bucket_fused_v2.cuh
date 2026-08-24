#pragma once

// Compatibility shim. The LOW center-partner fix is now integrated into
// bucket_low_orbit_kernel in ramstream32_bucket_fused.cuh. Keep the old entry
// point temporarily so existing probes/drivers remain source-compatible while
// compiling only one LOW device kernel.

static void bucket_run_low_fused_v2(
    const StorageLayout& layout,int threads=256,int grid_x=16,int grid_y=8
){
    bucket_run_low_fused(layout,threads,grid_x,grid_y);
}
