#pragma once

#ifndef P10DC_RANKFORMULA_QUAD_MLP
#define P10DC_RANKFORMULA_QUAD_MLP 0
#endif
#ifndef P10DC_RANKFORMULA_QUAD_OVERLAP_LOCAL
#define P10DC_RANKFORMULA_QUAD_OVERLAP_LOCAL 0
#endif
#if P10DC_RANKFORMULA_QUAD_MLP
#include "ramstream32_bucket_closure_pattern10_depthcode_delta_direct_affine_rankformula_nometa4_abstract_quad.cuh"
#else
#include "ramstream32_bucket_closure_pattern10_depthcode_delta_direct_affine_rankformula_nometa4_abstract.cuh"
#endif
#if P10DC_RANKFORMULA_QUAD_OVERLAP_LOCAL
#include "ramstream32_bucket_closure_pattern10_depthcode_rankformula_quad_overlap_local.cuh"
#endif

#ifndef P10DC_RANKFORMULA_PRECTX_FORWARD
#define P10DC_RANKFORMULA_PRECTX_FORWARD 0
#endif
#ifndef P10DC_RANKFORMULA_PRECTX_REVERSE
#define P10DC_RANKFORMULA_PRECTX_REVERSE 0
#endif
#ifndef P10DC_RANKFORMULA_PRECTX_COMPACT
#define P10DC_RANKFORMULA_PRECTX_COMPACT 0
#endif
#ifndef P10DC_ORBITCTA_FLAT_CHUNK
#define P10DC_ORBITCTA_FLAT_CHUNK 1
#endif
static_assert(P10DC_RANKFORMULA_PRECTX_COMPACT == 0 ||
              P10DC_RANKFORMULA_PRECTX_COMPACT == 1,
              "P10DC_RANKFORMULA_PRECTX_COMPACT must be 0 or 1");
static_assert(!P10DC_RANKFORMULA_PRECTX_COMPACT ||
              P10DC_RANKFORMULA_PRECTX_FORWARD ||
              P10DC_RANKFORMULA_PRECTX_REVERSE,
              "compact prectx requires forward and/or reverse prectx");
static_assert(P10DC_ORBITCTA_FLAT_CHUNK == 1 ||
              P10DC_ORBITCTA_FLAT_CHUNK == 2 ||
              P10DC_ORBITCTA_FLAT_CHUNK == 4 ||
              P10DC_ORBITCTA_FLAT_CHUNK == 8 ||
              P10DC_ORBITCTA_FLAT_CHUNK == 16 ||
              P10DC_ORBITCTA_FLAT_CHUNK == 32,
              "P10DC_ORBITCTA_FLAT_CHUNK must be 1,2,4,8,16,32");
static_assert(!P10DC_RANKFORMULA_QUAD_OVERLAP_LOCAL || P10DC_RANKFORMULA_QUAD_MLP,
              "flat quad overlap-local requires QUAD_MLP");
#if P10DC_RANKFORMULA_QUAD_MLP && P10DC_ORBITCTA_FLAT_CHUNK > 1
static_assert(P10DC_ORBITCTA_COL_ILP == 4,
              "chunked flat quad path requires ORBITCTA_COL_ILP=4");
#endif
#if P10DC_RANKFORMULA_PRECTX_FORWARD || P10DC_RANKFORMULA_PRECTX_REVERSE
#if P10DC_RANKFORMULA_PRECTX_COMPACT
#include "ramstream32_bucket_precomputed_high_ctx_compact.cuh"
#else
#include "ramstream32_bucket_precomputed_forward_high_ctx.cuh"
#endif
#endif

#if P10DC_RANKFORMULA_GATHER_MLP
#define P10DC_ORBITCTA_EARLY_JP 1
#define P10DC_ORBITCTA_EARLY_JP_LOCAL 1
#endif
#define P10DC_ORBITCTA_CTX P10DCDirectHighResolvedCtx
// The flat scheduler calls the global stream index q rather than qi. Keep the
// common prectx ABI by binding the flat stream index directly here.
#if P10DC_RANKFORMULA_PRECTX_FORWARD
#define P10DC_ORBITCTA_PREPARE_FORWARD(c,payload,loc,p,ss,js,ds,sr,jr,dr) do { \
    (void)(payload); (void)(loc); (void)(p); \
    p10dc_direct_resolve_high_io((c),(ss),(js),(ds),(sr),(jr),(dr)); \
    p10dc_apply_forward_prectx((c), q, nn); \
} while(0)
#else
#define P10DC_ORBITCTA_PREPARE_FORWARD(c,payload,loc,p,ss,js,ds,sr,jr,dr) \
    p10dc_prepare_forward_high_delta_direct_affine((c),(payload),(loc),(p),(ss),(js),(ds),(sr),(jr),(dr))
#endif
#if P10DC_RANKFORMULA_PRECTX_REVERSE
#define P10DC_ORBITCTA_PREPARE_REVERSE(c,payload,loc,plan_db,p,edge,ss,js,ds,sr,jr,dr) do { \
    (void)(payload); (void)(loc); (void)(plan_db); (void)(p); (void)(edge); \
    p10dc_direct_resolve_high_io((c),(ss),(js),(ds),(sr),(jr),(dr)); \
    p10dc_apply_reverse_prectx((c), q, kind); \
} while(0)
#else
#define P10DC_ORBITCTA_PREPARE_REVERSE(c,payload,loc,plan_db,p,edge,ss,js,ds,sr,jr,dr) \
    p10dc_prepare_reverse_high_delta_direct_affine((c),(payload),(loc),(plan_db),(p),(edge),(ss),(js),(ds),(sr),(jr),(dr))
#endif
#define P10DC_ORBITCTA_PLAN_SUM(c,db,lr) \
    p10dc_direct_resolved_high_plan_sum_cross5_rankformula_nometa4_abstract((c),(db),(lr))
#if P10DC_RANKFORMULA_PAIR_MLP
#define P10DC_ORBITCTA_PLAN_SUM_PAIR(c,db,lr0,lr1,out0,out1) \
    p10dc_direct_resolved_high_plan_sum_pair_cross5_rankformula_nometa4_abstract( \
        (c),(db),(lr0),(lr1),(out0),(out1))
#define P10DC_ORBITCTA_PLAN_SUM_PAIR_LOCAL 1
#endif
#if P10DC_RANKFORMULA_QUAD_MLP
#if P10DC_RANKFORMULA_QUAD_OVERLAP_LOCAL
#define P10DC_ORBITCTA_PLAN_SUM_QUAD(c,db,lr0,lr1,lr2,lr3,out0,out1,out2,out3) \
    p10dc_direct_resolved_high_plan_sum_quad_overlap_local( \
        (c),(db),(lr0),(lr1),(lr2),(lr3),(out0),(out1),(out2),(out3))
#else
#define P10DC_ORBITCTA_PLAN_SUM_QUAD(c,db,lr0,lr1,lr2,lr3,out0,out1,out2,out3) \
    p10dc_direct_resolved_high_plan_sum_quad_cross5_rankformula_nometa4_abstract( \
        (c),(db),(lr0),(lr1),(lr2),(lr3),(out0),(out1),(out2),(out3))
#endif
#define P10DC_ORBITCTA_PLAN_SUM_QUAD_LOCAL 1
#endif

#if P10DC_RANKFORMULA_QUAD_MLP && P10DC_ORBITCTA_FLAT_CHUNK > 1
// Preserve the ordinary flat kernel with its proven pair executor. The chunked
// kernels are compiled after the rename is removed, so they see the quad-aware
// executors below without changing the non-chunked flat path.
#define p10dc_orbitcta_flat_forward_columns p10dc_orbitcta_flat_forward_columns_pair_base
#define p10dc_orbitcta_flat_reverse_columns p10dc_orbitcta_flat_reverse_columns_pair_base
#include "ramstream32_bucket_orbit_closure_pattern10_depthcode_orbitcta_flat.cuh"
#undef p10dc_orbitcta_flat_reverse_columns
#undef p10dc_orbitcta_flat_forward_columns
#include "ramstream32_bucket_orbit_closure_pattern10_depthcode_orbitcta_flat_quad_columns.cuh"
#include "ramstream32_bucket_orbit_closure_pattern10_depthcode_orbitcta_flat_chunked.cuh"
#else
#include "ramstream32_bucket_orbit_closure_pattern10_depthcode_orbitcta_flat.cuh"
#if P10DC_ORBITCTA_FLAT_CHUNK > 1
#include "ramstream32_bucket_orbit_closure_pattern10_depthcode_orbitcta_flat_chunked.cuh"
#endif
#endif

#ifdef P10DC_ORBITCTA_PLAN_SUM_QUAD_LOCAL
#undef P10DC_ORBITCTA_PLAN_SUM_QUAD_LOCAL
#undef P10DC_ORBITCTA_PLAN_SUM_QUAD
#endif
#ifdef P10DC_ORBITCTA_PLAN_SUM_PAIR_LOCAL
#undef P10DC_ORBITCTA_PLAN_SUM_PAIR_LOCAL
#undef P10DC_ORBITCTA_PLAN_SUM_PAIR
#endif
#undef P10DC_ORBITCTA_PLAN_SUM
#undef P10DC_ORBITCTA_PREPARE_REVERSE
#undef P10DC_ORBITCTA_PREPARE_FORWARD
#undef P10DC_ORBITCTA_CTX
#ifdef P10DC_ORBITCTA_EARLY_JP_LOCAL
#undef P10DC_ORBITCTA_EARLY_JP_LOCAL
#undef P10DC_ORBITCTA_EARLY_JP
#endif
