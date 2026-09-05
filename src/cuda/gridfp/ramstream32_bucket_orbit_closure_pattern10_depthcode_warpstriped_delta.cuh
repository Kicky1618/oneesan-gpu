#pragma once

#include "ramstream32_bucket_closure_pattern10_depthcode_delta_plan.cuh"

// Reuse the proven warp-striped scheduling kernel, replacing only HIGH closure
// plan reconstruction with the ternary-delta implementation. LOW remains on
// the canonical depthcode path so the experiment isolates the HIGH hot path.
#define p10dc_forward_high p10dc_forward_high_delta
#define p10dc_reverse_high p10dc_reverse_high_delta
#define bucket_high_orbit_closure_pattern10_depthcode_warpstriped_kernel \
    bucket_high_orbit_closure_pattern10_depthcode_warpstriped_delta_kernel
#define bucket_reverse_high_pattern10_depthcode_warpstriped_kernel \
    bucket_reverse_high_pattern10_depthcode_warpstriped_delta_kernel
#include "ramstream32_bucket_orbit_closure_pattern10_depthcode_warpstriped.cuh"
#undef bucket_reverse_high_pattern10_depthcode_warpstriped_kernel
#undef bucket_high_orbit_closure_pattern10_depthcode_warpstriped_kernel
#undef p10dc_reverse_high
#undef p10dc_forward_high
