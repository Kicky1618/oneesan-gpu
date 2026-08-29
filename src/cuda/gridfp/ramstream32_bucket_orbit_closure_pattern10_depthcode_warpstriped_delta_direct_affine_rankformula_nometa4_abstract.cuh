#pragma once

// Pair/overlap helpers directly read SPARSE64 index/primary pointers.  Make the
// runtime declarations visible before those helpers are parsed; the B300 table
// builder itself is included later by the translation-unit entry point.
#if defined(P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64) && P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64
#include "ramstream32_bucket_closure_cross5_rankformula_nometa4_directgather64.cuh"
#include "ramstream32_bucket_closure_cross5_rankformula_nometa4_directgather_sparse64.cuh"
#endif

#ifndef P10DC_RANKFORMULA_QUAD_MLP
#define P10DC_RANKFORMULA_QUAD_MLP 0
#endif
#if P10DC_RANKFORMULA_QUAD_MLP
#include "ramstream32_bucket_closure_pattern10_depthcode_delta_direct_affine_rankformula_nometa4_abstract_quad.cuh"
#else
#include "ramstream32_bucket_closure_pattern10_depthcode_delta_direct_affine_rankformula_nometa4_abstract.cuh"
#endif
#include "ramstream32_bucket_rankformula_high_plan_profile.cuh"

#ifndef P10DC_RANKFORMULA_PRECTX_FORWARD
#define P10DC_RANKFORMULA_PRECTX_FORWARD 0
#endif
#ifndef P10DC_RANKFORMULA_PRECTX_REVERSE
#define P10DC_RANKFORMULA_PRECTX_REVERSE 0
#endif
#ifndef P10DC_RANKFORMULA_CPASYNC_LOCAL_PAIR
#define P10DC_RANKFORMULA_CPASYNC_LOCAL_PAIR 0
#endif
#ifndef P10DC_RANKFORMULA_CPASYNC_OVERLAP_LOCAL_PAIR
#define P10DC_RANKFORMULA_CPASYNC_OVERLAP_LOCAL_PAIR 0
#endif
#ifndef P10DC_RANKFORMULA_CPASYNC_OVERLAP_LOCAL_PIPE2
#define P10DC_RANKFORMULA_CPASYNC_OVERLAP_LOCAL_PIPE2 0
#endif
static_assert(P10DC_RANKFORMULA_CPASYNC_LOCAL_PAIR == 0 ||
              P10DC_RANKFORMULA_CPASYNC_LOCAL_PAIR == 1,
              "P10DC_RANKFORMULA_CPASYNC_LOCAL_PAIR must be 0 or 1");
static_assert(P10DC_RANKFORMULA_CPASYNC_OVERLAP_LOCAL_PAIR == 0 ||
              P10DC_RANKFORMULA_CPASYNC_OVERLAP_LOCAL_PAIR == 1,
              "P10DC_RANKFORMULA_CPASYNC_OVERLAP_LOCAL_PAIR must be 0 or 1");
static_assert(P10DC_RANKFORMULA_CPASYNC_OVERLAP_LOCAL_PIPE2 == 0 ||
              P10DC_RANKFORMULA_CPASYNC_OVERLAP_LOCAL_PIPE2 == 1,
              "P10DC_RANKFORMULA_CPASYNC_OVERLAP_LOCAL_PIPE2 must be 0 or 1");
static_assert(!(P10DC_RANKFORMULA_CPASYNC_LOCAL_PAIR &&
                P10DC_RANKFORMULA_CPASYNC_OVERLAP_LOCAL_PAIR),
              "local-cpasync and overlap-local pair modes are isolated A/B paths");
static_assert(!P10DC_RANKFORMULA_CPASYNC_OVERLAP_LOCAL_PAIR ||
              !P10DC_RANKFORMULA_QUAD_MLP,
              "overlap-local pair is isolated from QUAD_MLP");
static_assert(!P10DC_RANKFORMULA_CPASYNC_OVERLAP_LOCAL_PIPE2 ||
              P10DC_RANKFORMULA_CPASYNC_OVERLAP_LOCAL_PAIR,
              "overlap-local pipe2 requires overlap-local pair mode");
#if P10DC_RANKFORMULA_PRECTX_FORWARD || P10DC_RANKFORMULA_PRECTX_REVERSE
#include "ramstream32_bucket_precomputed_forward_high_ctx.cuh"
#endif
#if P10DC_RANKFORMULA_CPASYNC_LOCAL_PAIR
#include "ramstream32_bucket_closure_pattern10_depthcode_rankformula_local_cpasync_pair.cuh"
#endif
#if P10DC_RANKFORMULA_CPASYNC_OVERLAP_LOCAL_PAIR
#include "ramstream32_bucket_closure_pattern10_depthcode_rankformula_overlap_local_pair.cuh"
#endif
#if P10DC_RANKFORMULA_CPASYNC_OVERLAP_LOCAL_PIPE2
#include "ramstream32_bucket_closure_pattern10_depthcode_rankformula_overlap_local_pair_pipe2.cuh"
#endif

#if P10DC_RANKFORMULA_GATHER_MLP
#define P10DC_WARPSTRIPED_EARLY_JP_LOAD 1
#define P10DC_WARPSTRIPED_EARLY_JP_LOAD_LOCAL 1
#endif
#define P10DC_WARPSTRIPED_CTX P10DCDirectHighResolvedCtx
#if P10DC_RANKFORMULA_PRECTX_FORWARD
#define P10DC_WARPSTRIPED_PREPARE_FORWARD(c,payload,loc,p,ss,js,ds,sr,jr,dr) do { \
    (void)(payload); (void)(loc); (void)(p); \
    p10dc_direct_resolve_high_io((c),(ss),(js),(ds),(sr),(jr),(dr)); \
    p10dc_apply_forward_prectx((c), qi, nn); \
    p10dc_rankformula_profile_high_plan((c), 0u); \
} while(0)
#else
#define P10DC_WARPSTRIPED_PREPARE_FORWARD(c,payload,loc,p,ss,js,ds,sr,jr,dr) do { \
    p10dc_prepare_forward_high_delta_direct_affine((c),(payload),(loc),(p),(ss),(js),(ds),(sr),(jr),(dr)); \
    p10dc_rankformula_profile_high_plan((c), 0u); \
} while(0)
#endif
#if P10DC_RANKFORMULA_PRECTX_REVERSE
#define P10DC_WARPSTRIPED_PREPARE_REVERSE(c,payload,loc,plan_db,p,edge,ss,js,ds,sr,jr,dr) do { \
    (void)(payload); (void)(loc); (void)(plan_db); (void)(p); (void)(edge); \
    p10dc_direct_resolve_high_io((c),(ss),(js),(ds),(sr),(jr),(dr)); \
    p10dc_apply_reverse_prectx((c), qi, kind); \
    p10dc_rankformula_profile_high_plan((c), 1u); \
} while(0)
#else
#define P10DC_WARPSTRIPED_PREPARE_REVERSE(c,payload,loc,plan_db,p,edge,ss,js,ds,sr,jr,dr) do { \
    p10dc_prepare_reverse_high_delta_direct_affine((c),(payload),(loc),(plan_db),(p),(edge),(ss),(js),(ds),(sr),(jr),(dr)); \
    p10dc_rankformula_profile_high_plan((c), 1u); \
} while(0)
#endif
#define p10dc_resolved_high_plan_sum p10dc_direct_resolved_high_plan_sum_cross5_rankformula_nometa4_abstract
#if P10DC_RANKFORMULA_PAIR_MLP
static_assert(P10DC_WARPSTRIPED_COL_ILP == 2 || P10DC_WARPSTRIPED_COL_ILP == 4,
              "PAIR_MLP requires COL_ILP=2 or 4");
#if P10DC_RANKFORMULA_CPASYNC_OVERLAP_LOCAL_PIPE2
#define P10DC_WARPSTRIPED_PLAN_SUM_PAIR(c,db,lr0,lr1,out0,out1) \
    p10dc_direct_resolved_high_plan_sum_pair_overlap_local_pipe2( \
        (c),(db),(lr0),(lr1),(out0),(out1))
#elif P10DC_RANKFORMULA_CPASYNC_OVERLAP_LOCAL_PAIR
#define P10DC_WARPSTRIPED_PLAN_SUM_PAIR(c,db,lr0,lr1,out0,out1) \
    p10dc_direct_resolved_high_plan_sum_pair_overlap_local( \
        (c),(db),(lr0),(lr1),(out0),(out1))
#elif P10DC_RANKFORMULA_CPASYNC_LOCAL_PAIR
#define P10DC_WARPSTRIPED_PLAN_SUM_PAIR(c,db,lr0,lr1,out0,out1) \
    p10dc_direct_resolved_high_plan_sum_pair_local_cpasync( \
        (c),(db),(lr0),(lr1),(out0),(out1))
#else
#define P10DC_WARPSTRIPED_PLAN_SUM_PAIR(c,db,lr0,lr1,out0,out1) \
    p10dc_direct_resolved_high_plan_sum_pair_cross5_rankformula_nometa4_abstract( \
        (c),(db),(lr0),(lr1),(out0),(out1))
#endif
#define P10DC_WARPSTRIPED_PLAN_SUM_PAIR_LOCAL 1
#endif
#if P10DC_RANKFORMULA_QUAD_MLP
static_assert(P10DC_WARPSTRIPED_COL_ILP == 4,
              "QUAD_MLP requires COL_ILP=4");
#define P10DC_WARPSTRIPED_PLAN_SUM_QUAD(c,db,lr0,lr1,lr2,lr3,out0,out1,out2,out3) \
    p10dc_direct_resolved_high_plan_sum_quad_cross5_rankformula_nometa4_abstract( \
        (c),(db),(lr0),(lr1),(lr2),(lr3),(out0),(out1),(out2),(out3))
#define P10DC_WARPSTRIPED_PLAN_SUM_QUAD_LOCAL 1
#endif
#define bucket_high_orbit_closure_pattern10_depthcode_warpstriped_kernel \
    bucket_high_orbit_closure_pattern10_depthcode_warpstriped_delta_direct_affine_rankformula_nometa4_abstract_kernel
#define bucket_reverse_high_pattern10_depthcode_warpstriped_kernel \
    bucket_reverse_high_pattern10_depthcode_warpstriped_delta_direct_affine_rankformula_nometa4_abstract_kernel
#include "ramstream32_bucket_orbit_closure_pattern10_depthcode_warpstriped.cuh"
#undef bucket_reverse_high_pattern10_depthcode_warpstriped_kernel
#undef bucket_high_orbit_closure_pattern10_depthcode_warpstriped_kernel
#ifdef P10DC_WARPSTRIPED_PLAN_SUM_QUAD_LOCAL
#undef P10DC_WARPSTRIPED_PLAN_SUM_QUAD_LOCAL
#undef P10DC_WARPSTRIPED_PLAN_SUM_QUAD
#endif
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
