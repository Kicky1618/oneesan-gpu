#pragma once

#include "ramstream32_bucket_closure_pattern10_depthcode_delta_direct_affine_rankformula_nometa4_abstract_quad.cuh"
#if P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64
#include "ramstream32_bucket_closure_cross5_rankformula_nometa4_directgather_sparse64.cuh"
#endif

#ifndef P10DC_RANKFORMULA_QUAD_OVERLAP_LOCAL
#define P10DC_RANKFORMULA_QUAD_OVERLAP_LOCAL 0
#endif
#ifndef P10DC_RANKFORMULA_QUAD_SPARSE_DESC_MLP
#define P10DC_RANKFORMULA_QUAD_SPARSE_DESC_MLP 0
#endif
static_assert(P10DC_RANKFORMULA_QUAD_OVERLAP_LOCAL == 0 ||
              P10DC_RANKFORMULA_QUAD_OVERLAP_LOCAL == 1,
              "P10DC_RANKFORMULA_QUAD_OVERLAP_LOCAL must be 0 or 1");
static_assert(P10DC_RANKFORMULA_QUAD_SPARSE_DESC_MLP == 0 ||
              P10DC_RANKFORMULA_QUAD_SPARSE_DESC_MLP == 1,
              "P10DC_RANKFORMULA_QUAD_SPARSE_DESC_MLP must be 0 or 1");

#if P10DC_RANKFORMULA_QUAD_OVERLAP_LOCAL
static_assert(P10DC_RANKFORMULA_QUAD_MLP,
              "quad overlap-local requires QUAD_MLP");
static_assert(P10DC_RANKFORMULA_CPASYNC_PAIR,
              "quad overlap-local requires cp.async source staging");
static_assert(P10DC_RANKFORMULA_DIRECTGATHER64,
              "quad overlap-local requires DIRECTGATHER64");
static_assert(P10DC_RANKFORMULA_CPASYNC_VALUES_PER_THREAD >= 28,
              "quad overlap-local requires 28 shared source slots per thread");
#if P10DC_RANKFORMULA_QUAD_SPARSE_DESC_MLP
static_assert(P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64,
              "quad sparse descriptor MLP requires SPARSE64");
#endif

struct P10DCDirectGatherQuadPrimary {
    P10DCDirectGather64Word p0=0,p1=0,p2=0,p3=0;
};

__device__ __forceinline__ P10DCDirectGather64Word
p10dc_rankformula_qol_directgather64_primary_at(size_t gi) {
#if P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64
    const size_t wi=gi>>5;
    const uint32_t bit=uint32_t(gi)&31u;
    const P10DCDirectGather64Word ix=__ldg(
        D_P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64_INDEX+wi);
    const uint32_t bits=uint32_t(ix),flag=1u<<bit;
    if(!(bits&flag)) return 0;
    const uint32_t lower=bit?(bits&(flag-1u)):0u;
    const uint32_t ci=uint32_t(ix>>32)+uint32_t(__popc(lower));
    return __ldg(D_P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64_PRIMARY+ci);
#else
    return __ldg(D_P10DC_RANKFORMULA_DIRECTGATHER64+gi);
#endif
}

__device__ __forceinline__ P10DCDirectGatherQuadPrimary
p10dc_rankformula_qol_directgather64_quad_primary(
    uint32_t h,uint32_t rank0,uint32_t rank1,
    uint32_t rank2,uint32_t rank3,uint32_t depth
) {
    const size_t gi0=p10dc_rankformula_directgather_index(h,rank0,depth);
    const size_t gi1=p10dc_rankformula_directgather_index(h,rank1,depth);
    const size_t gi2=p10dc_rankformula_directgather_index(h,rank2,depth);
    const size_t gi3=p10dc_rankformula_directgather_index(h,rank3,depth);
    P10DCDirectGatherQuadPrimary z{};
#if P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64 && P10DC_RANKFORMULA_QUAD_SPARSE_DESC_MLP
    // Preserve four-column descriptor MLP without reviving the old 28-rank
    // register footprint. First put all four sparse index words in flight.
    const size_t wi0=gi0>>5,wi1=gi1>>5,wi2=gi2>>5,wi3=gi3>>5;
    const uint32_t bit0=uint32_t(gi0)&31u,bit1=uint32_t(gi1)&31u;
    const uint32_t bit2=uint32_t(gi2)&31u,bit3=uint32_t(gi3)&31u;
    const P10DCDirectGather64Word ix0=__ldg(D_P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64_INDEX+wi0);
    const P10DCDirectGather64Word ix1=__ldg(D_P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64_INDEX+wi1);
    const P10DCDirectGather64Word ix2=__ldg(D_P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64_INDEX+wi2);
    const P10DCDirectGather64Word ix3=__ldg(D_P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64_INDEX+wi3);

    const uint32_t bits0=uint32_t(ix0),bits1=uint32_t(ix1);
    const uint32_t bits2=uint32_t(ix2),bits3=uint32_t(ix3);
    const uint32_t flag0=1u<<bit0,flag1=1u<<bit1,flag2=1u<<bit2,flag3=1u<<bit3;
    const bool hit0=(bits0&flag0)!=0u,hit1=(bits1&flag1)!=0u;
    const bool hit2=(bits2&flag2)!=0u,hit3=(bits3&flag3)!=0u;
    uint32_t ci0=0,ci1=0,ci2=0,ci3=0;
    if(hit0){const uint32_t lo=bit0?(bits0&(flag0-1u)):0u;ci0=uint32_t(ix0>>32)+uint32_t(__popc(lo));}
    if(hit1){const uint32_t lo=bit1?(bits1&(flag1-1u)):0u;ci1=uint32_t(ix1>>32)+uint32_t(__popc(lo));}
    if(hit2){const uint32_t lo=bit2?(bits2&(flag2-1u)):0u;ci2=uint32_t(ix2>>32)+uint32_t(__popc(lo));}
    if(hit3){const uint32_t lo=bit3?(bits3&(flag3-1u)):0u;ci3=uint32_t(ix3>>32)+uint32_t(__popc(lo));}

    // Compact primary loads are also independent. Keep them adjacent so the
    // compiler can overlap the four memory dependencies before rank decode.
    if(hit0) z.p0=__ldg(D_P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64_PRIMARY+ci0);
    if(hit1) z.p1=__ldg(D_P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64_PRIMARY+ci1);
    if(hit2) z.p2=__ldg(D_P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64_PRIMARY+ci2);
    if(hit3) z.p3=__ldg(D_P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64_PRIMARY+ci3);
#else
    z.p0=p10dc_rankformula_qol_directgather64_primary_at(gi0);
    z.p1=p10dc_rankformula_qol_directgather64_primary_at(gi1);
    z.p2=p10dc_rankformula_qol_directgather64_primary_at(gi2);
    z.p3=p10dc_rankformula_qol_directgather64_primary_at(gi3);
#endif
    return z;
}

__device__ __forceinline__ uint32_t
p10dc_rankformula_qol_directgather64_count(P10DCDirectGather64Word p) {
    return uint32_t((p>>45)&7u);
}

__device__ __forceinline__ void
p10dc_rankformula_quad_issue_primary_group(
    P10DCDirectGather64Word p,uint32_t slot,const Count* source_row
) {
    const uint32_t n=p10dc_rankformula_qol_directgather64_count(p);
    uint32_t r0=uint32_t(p&0x7fffu);
    uint32_t r1=uint32_t((p>>15)&0x7fffu);
    uint32_t r2=uint32_t((p>>30)&0x7fffu);
    uint32_t r3=0,r4=0,r5=0,r6=0;
    if(n>3u){
#if P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64
        const P10DCDirectGather64Word q=__ldg(
            D_P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64_RARE+uint32_t(p>>48));
#else
        const P10DCDirectGather64Word q=__ldg(
            D_P10DC_RANKFORMULA_DIRECTGATHER64_RARE+uint32_t(p>>48));
#endif
        r3=uint32_t(q&0x7fffu);r4=uint32_t((q>>15)&0x7fffu);
        r5=uint32_t((q>>30)&0x7fffu);r6=uint32_t((q>>45)&0x7fffu);
    }
#define P10DC_QO_ISSUE(i,r,need) \
    p10dc_rankformula_cpasync_u32( \
        p10dc_rankformula_cpasync_slot(slot+(i)),source_row+(r),n>(need))
    P10DC_QO_ISSUE(0,r0,0);P10DC_QO_ISSUE(1,r1,1);
    P10DC_QO_ISSUE(2,r2,2);P10DC_QO_ISSUE(3,r3,3);
    P10DC_QO_ISSUE(4,r4,4);P10DC_QO_ISSUE(5,r5,5);
    P10DC_QO_ISSUE(6,r6,6);
#undef P10DC_QO_ISSUE
    p10dc_rankformula_cpasync_commit();
}

__device__ __forceinline__ BkczCrossAccum
p10dc_rankformula_quad_reduce_group(uint32_t base) {
#define P10DC_QO_SLOT(i) BkczCrossAccum(*p10dc_rankformula_cpasync_slot((i)))
    const BkczCrossAccum a01=p10dc_rankformula_accum_add(
        P10DC_QO_SLOT(base),P10DC_QO_SLOT(base+1));
    const BkczCrossAccum a23=p10dc_rankformula_accum_add(
        P10DC_QO_SLOT(base+2),P10DC_QO_SLOT(base+3));
    const BkczCrossAccum a45=p10dc_rankformula_accum_add(
        P10DC_QO_SLOT(base+4),P10DC_QO_SLOT(base+5));
    const BkczCrossAccum z=p10dc_rankformula_accum_add(
        p10dc_rankformula_accum_add(a01,a23),
        p10dc_rankformula_accum_add(a45,P10DC_QO_SLOT(base+6)));
#undef P10DC_QO_SLOT
    return z;
}

__device__ __forceinline__ void
p10dc_direct_resolved_high_plan_sum_quad_overlap_local(
    const P10DCDirectHighResolvedCtx& c,const BucketPhysicalBlock& db,
    uint32_t lr0,uint32_t lr1,uint32_t lr2,uint32_t lr3,
    Count& out0,Count& out1,Count& out2,Count& out3
) {
    P10DCDirectGatherQuadPrimary p{};
    if(c.cross_depth)
        p=p10dc_rankformula_qol_directgather64_quad_primary(
            db.hs,lr0,lr1,lr2,lr3,c.cross_depth);
    const uint32_t any=
        p10dc_rankformula_qol_directgather64_count(p.p0)|
        p10dc_rankformula_qol_directgather64_count(p.p1)|
        p10dc_rankformula_qol_directgather64_count(p.p2)|
        p10dc_rankformula_qol_directgather64_count(p.p3);
    if(any){
        // Decode one packed descriptor at a time and immediately launch its
        // source group. Rank temporaries die before the next column is decoded;
        // the source requests remain outstanding in the async pipeline.
        p10dc_rankformula_quad_issue_primary_group(p.p0,0,c.cross_base);
        p10dc_rankformula_quad_issue_primary_group(p.p1,7,c.cross_base);
        p10dc_rankformula_quad_issue_primary_group(p.p2,14,c.cross_base);
        p10dc_rankformula_quad_issue_primary_group(p.p3,21,c.cross_base);
    }

    // Hide CROSS latency with ordinary local-source work. The register helper
    // keeps only four values live per source row.
    BkczCrossAccum s0=0,s1=0,s2=0,s3=0;
    p10dc_rankformula_quad_local_register(c,lr0,lr1,lr2,lr3,s0,s1,s2,s3);

    if(any){
        p10dc_rankformula_cross_quad_wait3();
        s0=p10dc_rankformula_accum_add(s0,p10dc_rankformula_quad_reduce_group(0));
        p10dc_rankformula_cross_quad_wait2();
        s1=p10dc_rankformula_accum_add(s1,p10dc_rankformula_quad_reduce_group(7));
        p10dc_rankformula_cross_quad_wait1();
        s2=p10dc_rankformula_accum_add(s2,p10dc_rankformula_quad_reduce_group(14));
        p10dc_rankformula_cpasync_wait_all();
        s3=p10dc_rankformula_accum_add(s3,p10dc_rankformula_quad_reduce_group(21));
    }
#if GPU_DIRECT_PM_ACCUM
    out0=gpu_direct_pm_reduce_u64(s0);out1=gpu_direct_pm_reduce_u64(s1);
    out2=gpu_direct_pm_reduce_u64(s2);out3=gpu_direct_pm_reduce_u64(s3);
#else
    out0=Count(s0);out1=Count(s1);out2=Count(s2);out3=Count(s3);
#endif
}
#endif
