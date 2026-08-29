#pragma once

#include "ramstream32_bucket_closure_pattern10_depthcode_delta_direct_affine_rankformula_nometa4_abstract_quad.cuh"
#if P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64
#include "ramstream32_bucket_closure_cross5_rankformula_nometa4_directgather_sparse64.cuh"
#endif

#ifndef P10DC_RANKFORMULA_QUAD_OVERLAP_LOCAL
#define P10DC_RANKFORMULA_QUAD_OVERLAP_LOCAL 0
#endif
static_assert(P10DC_RANKFORMULA_QUAD_OVERLAP_LOCAL == 0 ||
              P10DC_RANKFORMULA_QUAD_OVERLAP_LOCAL == 1,
              "P10DC_RANKFORMULA_QUAD_OVERLAP_LOCAL must be 0 or 1");

#if P10DC_RANKFORMULA_QUAD_OVERLAP_LOCAL
static_assert(P10DC_RANKFORMULA_QUAD_MLP,
              "quad overlap-local requires QUAD_MLP");
static_assert(P10DC_RANKFORMULA_CPASYNC_PAIR,
              "quad overlap-local requires cp.async source staging");
static_assert(P10DC_RANKFORMULA_DIRECTGATHER64,
              "quad overlap-local requires DIRECTGATHER64");
static_assert(P10DC_RANKFORMULA_CPASYNC_VALUES_PER_THREAD >= 28,
              "quad overlap-local requires 28 shared source slots per thread");

struct P10DCDirectGatherQuadRanks {
    uint32_t n0=0,n1=0,n2=0,n3=0;
    uint32_t a0=0,a1=0,a2=0,a3=0,a4=0,a5=0,a6=0;
    uint32_t b0=0,b1=0,b2=0,b3=0,b4=0,b5=0,b6=0;
    uint32_t c0=0,c1=0,c2=0,c3=0,c4=0,c5=0,c6=0;
    uint32_t d0=0,d1=0,d2=0,d3=0,d4=0,d5=0,d6=0;
};

__device__ __forceinline__ P10DCDirectGatherQuadRanks
p10dc_rankformula_directgather64_quad_ranks(
    uint32_t h, uint32_t rank0, uint32_t rank1,
    uint32_t rank2, uint32_t rank3, uint32_t depth
) {
    P10DCDirectGatherQuadRanks z{};
    const size_t gi0=p10dc_rankformula_directgather_index(h,rank0,depth);
    const size_t gi1=p10dc_rankformula_directgather_index(h,rank1,depth);
    const size_t gi2=p10dc_rankformula_directgather_index(h,rank2,depth);
    const size_t gi3=p10dc_rankformula_directgather_index(h,rank3,depth);
    P10DCDirectGather64Word p0=0,p1=0,p2=0,p3=0;
#if P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64
    const size_t wi0=gi0>>5,wi1=gi1>>5,wi2=gi2>>5,wi3=gi3>>5;
    const uint32_t bit0=uint32_t(gi0)&31u,bit1=uint32_t(gi1)&31u;
    const uint32_t bit2=uint32_t(gi2)&31u,bit3=uint32_t(gi3)&31u;
    const P10DCDirectGather64Word ix0=__ldg(D_P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64_INDEX+wi0);
    const P10DCDirectGather64Word ix1=__ldg(D_P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64_INDEX+wi1);
    const P10DCDirectGather64Word ix2=__ldg(D_P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64_INDEX+wi2);
    const P10DCDirectGather64Word ix3=__ldg(D_P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64_INDEX+wi3);
    const uint32_t bits0=uint32_t(ix0),bits1=uint32_t(ix1),bits2=uint32_t(ix2),bits3=uint32_t(ix3);
    const uint32_t flag0=1u<<bit0,flag1=1u<<bit1,flag2=1u<<bit2,flag3=1u<<bit3;
    if(bits0&flag0){const uint32_t lower=bit0?(bits0&(flag0-1u)):0u;const uint32_t ci=uint32_t(ix0>>32)+uint32_t(__popc(lower));p0=__ldg(D_P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64_PRIMARY+ci);}
    if(bits1&flag1){const uint32_t lower=bit1?(bits1&(flag1-1u)):0u;const uint32_t ci=uint32_t(ix1>>32)+uint32_t(__popc(lower));p1=__ldg(D_P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64_PRIMARY+ci);}
    if(bits2&flag2){const uint32_t lower=bit2?(bits2&(flag2-1u)):0u;const uint32_t ci=uint32_t(ix2>>32)+uint32_t(__popc(lower));p2=__ldg(D_P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64_PRIMARY+ci);}
    if(bits3&flag3){const uint32_t lower=bit3?(bits3&(flag3-1u)):0u;const uint32_t ci=uint32_t(ix3>>32)+uint32_t(__popc(lower));p3=__ldg(D_P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64_PRIMARY+ci);}
#else
    p0=__ldg(D_P10DC_RANKFORMULA_DIRECTGATHER64+gi0);
    p1=__ldg(D_P10DC_RANKFORMULA_DIRECTGATHER64+gi1);
    p2=__ldg(D_P10DC_RANKFORMULA_DIRECTGATHER64+gi2);
    p3=__ldg(D_P10DC_RANKFORMULA_DIRECTGATHER64+gi3);
#endif
    z.n0=uint32_t((p0>>45)&7u);z.n1=uint32_t((p1>>45)&7u);
    z.n2=uint32_t((p2>>45)&7u);z.n3=uint32_t((p3>>45)&7u);
    z.a0=uint32_t(p0&0x7fffu);z.a1=uint32_t((p0>>15)&0x7fffu);z.a2=uint32_t((p0>>30)&0x7fffu);
    z.b0=uint32_t(p1&0x7fffu);z.b1=uint32_t((p1>>15)&0x7fffu);z.b2=uint32_t((p1>>30)&0x7fffu);
    z.c0=uint32_t(p2&0x7fffu);z.c1=uint32_t((p2>>15)&0x7fffu);z.c2=uint32_t((p2>>30)&0x7fffu);
    z.d0=uint32_t(p3&0x7fffu);z.d1=uint32_t((p3>>15)&0x7fffu);z.d2=uint32_t((p3>>30)&0x7fffu);
    P10DCDirectGather64Word q0=0,q1=0,q2=0,q3=0;
#if P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64
    if(z.n0>3)q0=__ldg(D_P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64_RARE+uint32_t(p0>>48));
    if(z.n1>3)q1=__ldg(D_P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64_RARE+uint32_t(p1>>48));
    if(z.n2>3)q2=__ldg(D_P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64_RARE+uint32_t(p2>>48));
    if(z.n3>3)q3=__ldg(D_P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64_RARE+uint32_t(p3>>48));
#else
    if(z.n0>3)q0=__ldg(D_P10DC_RANKFORMULA_DIRECTGATHER64_RARE+uint32_t(p0>>48));
    if(z.n1>3)q1=__ldg(D_P10DC_RANKFORMULA_DIRECTGATHER64_RARE+uint32_t(p1>>48));
    if(z.n2>3)q2=__ldg(D_P10DC_RANKFORMULA_DIRECTGATHER64_RARE+uint32_t(p2>>48));
    if(z.n3>3)q3=__ldg(D_P10DC_RANKFORMULA_DIRECTGATHER64_RARE+uint32_t(p3>>48));
#endif
    if(z.n0>3){z.a3=uint32_t(q0&0x7fffu);z.a4=uint32_t((q0>>15)&0x7fffu);z.a5=uint32_t((q0>>30)&0x7fffu);z.a6=uint32_t((q0>>45)&0x7fffu);}
    if(z.n1>3){z.b3=uint32_t(q1&0x7fffu);z.b4=uint32_t((q1>>15)&0x7fffu);z.b5=uint32_t((q1>>30)&0x7fffu);z.b6=uint32_t((q1>>45)&0x7fffu);}
    if(z.n2>3){z.c3=uint32_t(q2&0x7fffu);z.c4=uint32_t((q2>>15)&0x7fffu);z.c5=uint32_t((q2>>30)&0x7fffu);z.c6=uint32_t((q2>>45)&0x7fffu);}
    if(z.n3>3){z.d3=uint32_t(q3&0x7fffu);z.d4=uint32_t((q3>>15)&0x7fffu);z.d5=uint32_t((q3>>30)&0x7fffu);z.d6=uint32_t((q3>>45)&0x7fffu);}
    return z;
}

__device__ __forceinline__ void
p10dc_direct_resolved_high_plan_sum_quad_overlap_local(
    const P10DCDirectHighResolvedCtx& c,const BucketPhysicalBlock& db,
    uint32_t lr0,uint32_t lr1,uint32_t lr2,uint32_t lr3,
    Count& out0,Count& out1,Count& out2,Count& out3
) {
    P10DCDirectGatherQuadRanks r{};
    const bool have_cross=c.cross_depth!=0u;
    if(have_cross)r=p10dc_rankformula_directgather64_quad_ranks(
        db.hs,lr0,lr1,lr2,lr3,c.cross_depth);
    const bool pending=have_cross && (r.n0|r.n1|r.n2|r.n3);
    if(pending){
#define P10DC_QO_CPA(slot,rank,count,need) p10dc_rankformula_cpasync_u32(p10dc_rankformula_cpasync_slot(slot),c.cross_base+(rank),(count)>(need))
        P10DC_QO_CPA(0,r.a0,r.n0,0);P10DC_QO_CPA(1,r.a1,r.n0,1);P10DC_QO_CPA(2,r.a2,r.n0,2);P10DC_QO_CPA(3,r.a3,r.n0,3);P10DC_QO_CPA(4,r.a4,r.n0,4);P10DC_QO_CPA(5,r.a5,r.n0,5);P10DC_QO_CPA(6,r.a6,r.n0,6);p10dc_rankformula_cpasync_commit();
        P10DC_QO_CPA(7,r.b0,r.n1,0);P10DC_QO_CPA(8,r.b1,r.n1,1);P10DC_QO_CPA(9,r.b2,r.n1,2);P10DC_QO_CPA(10,r.b3,r.n1,3);P10DC_QO_CPA(11,r.b4,r.n1,4);P10DC_QO_CPA(12,r.b5,r.n1,5);P10DC_QO_CPA(13,r.b6,r.n1,6);p10dc_rankformula_cpasync_commit();
        P10DC_QO_CPA(14,r.c0,r.n2,0);P10DC_QO_CPA(15,r.c1,r.n2,1);P10DC_QO_CPA(16,r.c2,r.n2,2);P10DC_QO_CPA(17,r.c3,r.n2,3);P10DC_QO_CPA(18,r.c4,r.n2,4);P10DC_QO_CPA(19,r.c5,r.n2,5);P10DC_QO_CPA(20,r.c6,r.n2,6);p10dc_rankformula_cpasync_commit();
        P10DC_QO_CPA(21,r.d0,r.n3,0);P10DC_QO_CPA(22,r.d1,r.n3,1);P10DC_QO_CPA(23,r.d2,r.n3,2);P10DC_QO_CPA(24,r.d3,r.n3,3);P10DC_QO_CPA(25,r.d4,r.n3,4);P10DC_QO_CPA(26,r.d5,r.n3,5);P10DC_QO_CPA(27,r.d6,r.n3,6);p10dc_rankformula_cpasync_commit();
#undef P10DC_QO_CPA
    }

    // Local source work is deliberately performed while the four cross groups
    // are outstanding. Only four local values are live at a time, so this gets
    // overlap without the 32-value register explosion of a full preload.
    BkczCrossAccum s0=0,s1=0,s2=0,s3=0;
    p10dc_rankformula_quad_local_register(c,lr0,lr1,lr2,lr3,s0,s1,s2,s3);

    if(pending){
#define P10DC_QO_SLOT(i) BkczCrossAccum(*p10dc_rankformula_cpasync_slot(i))
#define P10DC_QO_REDUCE(base) p10dc_rankformula_accum_add( \
        p10dc_rankformula_accum_add( \
            p10dc_rankformula_accum_add(P10DC_QO_SLOT(base),P10DC_QO_SLOT((base)+1)), \
            p10dc_rankformula_accum_add(P10DC_QO_SLOT((base)+2),P10DC_QO_SLOT((base)+3))), \
        p10dc_rankformula_accum_add( \
            p10dc_rankformula_accum_add(P10DC_QO_SLOT((base)+4),P10DC_QO_SLOT((base)+5)),P10DC_QO_SLOT((base)+6)))
        p10dc_rankformula_cross_quad_wait3();s0=p10dc_rankformula_accum_add(s0,P10DC_QO_REDUCE(0));
        p10dc_rankformula_cross_quad_wait2();s1=p10dc_rankformula_accum_add(s1,P10DC_QO_REDUCE(7));
        p10dc_rankformula_cross_quad_wait1();s2=p10dc_rankformula_accum_add(s2,P10DC_QO_REDUCE(14));
        p10dc_rankformula_cpasync_wait_all();s3=p10dc_rankformula_accum_add(s3,P10DC_QO_REDUCE(21));
#undef P10DC_QO_REDUCE
#undef P10DC_QO_SLOT
    }
#if GPU_DIRECT_PM_ACCUM
    out0=gpu_direct_pm_reduce_u64(s0);out1=gpu_direct_pm_reduce_u64(s1);
    out2=gpu_direct_pm_reduce_u64(s2);out3=gpu_direct_pm_reduce_u64(s3);
#else
    out0=Count(s0);out1=Count(s1);out2=Count(s2);out3=Count(s3);
#endif
}
#endif
