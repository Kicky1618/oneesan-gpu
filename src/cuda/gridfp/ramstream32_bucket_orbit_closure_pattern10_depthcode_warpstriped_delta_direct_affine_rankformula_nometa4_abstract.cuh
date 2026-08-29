#pragma once

#include "ramstream32_bucket_closure_pattern10_depthcode_delta_direct_affine_rankformula_nometa4_abstract.cuh"

#ifndef P10DC_RANKFORMULA_PRECTX_FORWARD
#define P10DC_RANKFORMULA_PRECTX_FORWARD 0
#endif
#if P10DC_RANKFORMULA_PRECTX_FORWARD
#include "ramstream32_bucket_precomputed_forward_high_ctx.cuh"
#endif

#if P10DC_RANKFORMULA_GATHER_MLP
#define P10DC_WARPSTRIPED_EARLY_JP_LOAD 1
#define P10DC_WARPSTRIPED_EARLY_JP_LOAD_LOCAL 1
#endif
#define P10DC_WARPSTRIPED_CTX P10DCDirectHighResolvedCtx
#if P10DC_RANKFORMULA_PRECTX_FORWARD
// The generic warp-striped kernel has qi/nn in lexical scope here. Keep its
// scheduler fields intact; only replace the per-orbit execution context.
#define P10DC_WARPSTRIPED_PREPARE_FORWARD(c,payload,loc,p,ss,js,ds,sr,jr,dr) do { \
    (void)(payload); (void)(loc); (void)(p); (void)(ss); (void)(js); (void)(ds); \
    (void)(sr); (void)(jr); (void)(dr); \
    p10dc_apply_forward_prectx((c), qi, nn); \
} while(0)
#else
#define P10DC_WARPSTRIPED_PREPARE_FORWARD(c,payload,loc,p,ss,js,ds,sr,jr,dr) \
    p10dc_prepare_forward_high_delta_direct_affine((c),(payload),(loc),(p),(ss),(js),(ds),(sr),(jr),(dr))
#endif
#define P10DC_WARPSTRIPED_PREPARE_REVERSE(c,payload,loc,plan_db,p,edge,ss,js,ds,sr,jr,dr) \
    p10dc_prepare_reverse_high_delta_direct_affine((c),(payload),(loc),(plan_db),(p),(edge),(ss),(js),(ds),(sr),(jr),(dr))
#define p10dc_resolved_high_plan_sum p10dc_direct_resolved_high_plan_sum_cross5_rankformula_nometa4_abstract
#if P10DC_RANKFORMULA_PAIR_MLP
static_assert(P10DC_WARPSTRIPED_COL_ILP == 2 || P10DC_WARPSTRIPED_COL_ILP == 4,
              "PAIR_MLP requires COL_ILP=2 or 4");
#define P10DC_WARPSTRIPED_PLAN_SUM_PAIR(c,db,lr0,lr1,out0,out1) \
    p10dc_direct_resolved_high_plan_sum_pair_cross5_rankformula_nometa4_abstract( \
        (c),(db),(lr0),(lr1),(out0),(out1))
#define P10DC_WARPSTRIPED_PLAN_SUM_PAIR_LOCAL 1
#endif
#define bucket_high_orbit_closure_pattern10_depthcode_warpstriped_kernel \
    bucket_high_orbit_closure_pattern10_depthcode_warpstriped_delta_direct_affine_rankformula_nometa4_abstract_kernel
#define bucket_reverse_high_pattern10_depthcode_warpstriped_kernel \
    bucket_reverse_high_pattern10_depthcode_warpstriped_delta_direct_affine_rankformula_nometa4_abstract_kernel
#include "ramstream32_bucket_orbit_closure_pattern10_depthcode_warpstriped.cuh"
#undef bucket_reverse_high_pattern10_depthcode_warpstriped_kernel
#undef bucket_high_orbit_closure_pattern10_depthcode_warpstriped_kernel
#ifdef P10DC_WARPSTRIPED_PLAN_SUM_PAIR_LOCAL
#undef P10DC_WARPSTRIPED_PLAN_SUM_PAIR_LOCAL
#undef P10DC_WARPSTRIPED_PLAN_SUM_PAIR
#endif
#undef p10dc_resolved_high_plan_sum
#undef P10DC_WARPSTRIPED_PREPARE_REVERSE
#undef P10DC_WARPSTRIPED_PREPARE_FORWARD
#undef P10DC_WARPSTRIPED_CTX
#ifdef P10DC_WARPSTRIPED_EARLY_JP_LOAD_LOCAL
#undef P10DC_WARPSTRIPED_EARLY_JP_LOAD_LOCAL
#undef P10DC_WARPSTRIPED_EARLY_JP_LOAD
#endif
