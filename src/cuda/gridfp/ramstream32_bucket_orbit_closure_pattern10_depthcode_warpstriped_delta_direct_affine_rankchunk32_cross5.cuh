#pragma once

#ifndef P10DC_RANKCHUNK32_DIRECTMASK
#define P10DC_RANKCHUNK32_DIRECTMASK 0
#endif
static_assert(P10DC_RANKCHUNK32_DIRECTMASK == 0 || P10DC_RANKCHUNK32_DIRECTMASK == 1,
              "P10DC_RANKCHUNK32_DIRECTMASK must be 0 or 1");

#if P10DC_RANKCHUNK32_DIRECTMASK
#include "ramstream32_bucket_closure_pattern10_depthcode_delta_direct_affine_rankchunk32_directmask.cuh"
#else
#include "ramstream32_bucket_closure_pattern10_depthcode_delta_direct_affine_rankchunk32.cuh"
#endif

#define P10DC_WARPSTRIPED_CTX P10DCDirectHighResolvedCtx
#define P10DC_WARPSTRIPED_PREPARE_FORWARD(c,payload,loc,p,ss,js,ds,sr,jr,dr) \
    p10dc_prepare_forward_high_delta_direct_affine((c),(payload),(loc),(p),(ss),(js),(ds),(sr),(jr),(dr))
#define P10DC_WARPSTRIPED_PREPARE_REVERSE(c,payload,loc,plan_db,p,edge,ss,js,ds,sr,jr,dr) \
    p10dc_prepare_reverse_high_delta_direct_affine((c),(payload),(loc),(plan_db),(p),(edge),(ss),(js),(ds),(sr),(jr),(dr))
#if P10DC_RANKCHUNK32_DIRECTMASK
#define p10dc_resolved_high_plan_sum p10dc_direct_resolved_high_plan_sum_rankchunk32_directmask
#else
#define p10dc_resolved_high_plan_sum p10dc_direct_resolved_high_plan_sum_cross5_rankchunk32
#endif
#define bucket_high_orbit_closure_pattern10_depthcode_warpstriped_kernel \
    bucket_high_orbit_closure_pattern10_depthcode_warpstriped_delta_direct_affine_rankchunk32_cross5_kernel
#define bucket_reverse_high_pattern10_depthcode_warpstriped_kernel \
    bucket_reverse_high_pattern10_depthcode_warpstriped_delta_direct_affine_rankchunk32_cross5_kernel
#include "ramstream32_bucket_orbit_closure_pattern10_depthcode_warpstriped.cuh"
#undef bucket_reverse_high_pattern10_depthcode_warpstriped_kernel
#undef bucket_high_orbit_closure_pattern10_depthcode_warpstriped_kernel
#undef p10dc_resolved_high_plan_sum
#undef P10DC_WARPSTRIPED_PREPARE_REVERSE
#undef P10DC_WARPSTRIPED_PREPARE_FORWARD
#undef P10DC_WARPSTRIPED_CTX
