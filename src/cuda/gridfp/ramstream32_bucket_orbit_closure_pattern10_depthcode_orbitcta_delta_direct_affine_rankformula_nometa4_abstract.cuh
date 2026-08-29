#pragma once

#include "ramstream32_bucket_closure_pattern10_depthcode_delta_direct_affine_rankformula_nometa4_abstract.cuh"

#if P10DC_RANKFORMULA_GATHER_MLP
#define P10DC_ORBITCTA_EARLY_JP 1
#define P10DC_ORBITCTA_EARLY_JP_LOCAL 1
#endif
#define P10DC_ORBITCTA_CTX P10DCDirectHighResolvedCtx
#define P10DC_ORBITCTA_PREPARE_FORWARD(c,payload,loc,p,ss,js,ds,sr,jr,dr) \
    p10dc_prepare_forward_high_delta_direct_affine((c),(payload),(loc),(p),(ss),(js),(ds),(sr),(jr),(dr))
#define P10DC_ORBITCTA_PREPARE_REVERSE(c,payload,loc,plan_db,p,edge,ss,js,ds,sr,jr,dr) \
    p10dc_prepare_reverse_high_delta_direct_affine((c),(payload),(loc),(plan_db),(p),(edge),(ss),(js),(ds),(sr),(jr),(dr))
#define P10DC_ORBITCTA_PLAN_SUM(c,db,lr) \
    p10dc_direct_resolved_high_plan_sum_cross5_rankformula_nometa4_abstract((c),(db),(lr))
#if P10DC_RANKFORMULA_PAIR_MLP
#define P10DC_ORBITCTA_PLAN_SUM_PAIR(c,db,lr0,lr1,out0,out1) \
    p10dc_direct_resolved_high_plan_sum_pair_cross5_rankformula_nometa4_abstract( \
        (c),(db),(lr0),(lr1),(out0),(out1))
#define P10DC_ORBITCTA_PLAN_SUM_PAIR_LOCAL 1
#endif
#include "ramstream32_bucket_orbit_closure_pattern10_depthcode_orbitcta.cuh"
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
