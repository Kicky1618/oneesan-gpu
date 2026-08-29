#pragma once

#include "ramstream32_bucket_closure_cross5_rankformula_nometa4_directgather64_pair.cuh"

#ifndef P10DC_RANKFORMULA_QUAD_MLP
#define P10DC_RANKFORMULA_QUAD_MLP 0
#endif
static_assert(P10DC_RANKFORMULA_QUAD_MLP == 0 || P10DC_RANKFORMULA_QUAD_MLP == 1,
              "P10DC_RANKFORMULA_QUAD_MLP must be 0 or 1");

#if P10DC_RANKFORMULA_QUAD_MLP
static_assert(P10DC_RANKFORMULA_DIRECTGATHER64,
              "QUAD_MLP currently requires DIRECTGATHER64");
static_assert(P10DC_RANKFORMULA_PAIR_MLP,
              "QUAD_MLP uses the pair-MLP runtime contract");
#if P10DC_RANKFORMULA_CPASYNC_PAIR
static_assert(P10DC_RANKFORMULA_CPASYNC_VALUES_PER_THREAD >= 28,
              "cp.async QUAD_MLP requires 28 source slots per thread");
#endif

struct P10DCRankFormulaQuadAccum {
    BkczCrossAccum a, b, c, d;
};

__device__ __forceinline__ P10DCRankFormulaQuadAccum
p10dc_resolved_low_preimages_cross5_rankformula_nometa4_directgather64_quad_fixed(
    uint32_t h,
    uint32_t rank0, uint32_t rank1, uint32_t rank2, uint32_t rank3,
    uint32_t depth, const Count* source_row
) {
    if (!depth || depth > P10DC_RANKFORMULA_ABSTRACT_SELECT_DEPTHS)
        return P10DCRankFormulaQuadAccum{0, 0, 0, 0};

    const size_t gi0 = p10dc_rankformula_directgather_index(h, rank0, depth);
    const size_t gi1 = p10dc_rankformula_directgather_index(h, rank1, depth);
    const size_t gi2 = p10dc_rankformula_directgather_index(h, rank2, depth);
    const size_t gi3 = p10dc_rankformula_directgather_index(h, rank3, depth);
    P10DCDirectGather64Word p0 = 0, p1 = 0, p2 = 0, p3 = 0;

#if P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64
    const size_t wi0 = gi0 >> 5, wi1 = gi1 >> 5, wi2 = gi2 >> 5, wi3 = gi3 >> 5;
    const uint32_t bit0 = uint32_t(gi0) & 31u, bit1 = uint32_t(gi1) & 31u;
    const uint32_t bit2 = uint32_t(gi2) & 31u, bit3 = uint32_t(gi3) & 31u;
    const P10DCDirectGather64Word ix0 = __ldg(D_P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64_INDEX + wi0);
    const P10DCDirectGather64Word ix1 = __ldg(D_P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64_INDEX + wi1);
    const P10DCDirectGather64Word ix2 = __ldg(D_P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64_INDEX + wi2);
    const P10DCDirectGather64Word ix3 = __ldg(D_P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64_INDEX + wi3);
    const uint32_t bits0 = uint32_t(ix0), bits1 = uint32_t(ix1), bits2 = uint32_t(ix2), bits3 = uint32_t(ix3);
    const uint32_t flag0 = 1u << bit0, flag1 = 1u << bit1, flag2 = 1u << bit2, flag3 = 1u << bit3;
    const bool live0 = (bits0 & flag0) != 0u, live1 = (bits1 & flag1) != 0u;
    const bool live2 = (bits2 & flag2) != 0u, live3 = (bits3 & flag3) != 0u;
    if (!(live0 || live1 || live2 || live3)) return P10DCRankFormulaQuadAccum{0, 0, 0, 0};
    uint32_t ci0 = 0, ci1 = 0, ci2 = 0, ci3 = 0;
    if (live0) ci0 = uint32_t(ix0 >> 32) + uint32_t(__popc(bit0 ? (bits0 & (flag0 - 1u)) : 0u));
    if (live1) ci1 = uint32_t(ix1 >> 32) + uint32_t(__popc(bit1 ? (bits1 & (flag1 - 1u)) : 0u));
    if (live2) ci2 = uint32_t(ix2 >> 32) + uint32_t(__popc(bit2 ? (bits2 & (flag2 - 1u)) : 0u));
    if (live3) ci3 = uint32_t(ix3 >> 32) + uint32_t(__popc(bit3 ? (bits3 & (flag3 - 1u)) : 0u));
    if (live0) p0 = __ldg(D_P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64_PRIMARY + ci0);
    if (live1) p1 = __ldg(D_P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64_PRIMARY + ci1);
    if (live2) p2 = __ldg(D_P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64_PRIMARY + ci2);
    if (live3) p3 = __ldg(D_P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64_PRIMARY + ci3);
#else
    p0 = __ldg(D_P10DC_RANKFORMULA_DIRECTGATHER64 + gi0);
    p1 = __ldg(D_P10DC_RANKFORMULA_DIRECTGATHER64 + gi1);
    p2 = __ldg(D_P10DC_RANKFORMULA_DIRECTGATHER64 + gi2);
    p3 = __ldg(D_P10DC_RANKFORMULA_DIRECTGATHER64 + gi3);
#endif

    const uint32_t n0 = uint32_t((p0 >> 45) & 7u), n1 = uint32_t((p1 >> 45) & 7u);
    const uint32_t n2 = uint32_t((p2 >> 45) & 7u), n3 = uint32_t((p3 >> 45) & 7u);
    if (!(n0 | n1 | n2 | n3)) return P10DCRankFormulaQuadAccum{0, 0, 0, 0};

    uint32_t a0=uint32_t(p0&0x7fffu),a1=uint32_t((p0>>15)&0x7fffu),a2=uint32_t((p0>>30)&0x7fffu),a3=0,a4=0,a5=0,a6=0;
    uint32_t b0=uint32_t(p1&0x7fffu),b1=uint32_t((p1>>15)&0x7fffu),b2=uint32_t((p1>>30)&0x7fffu),b3=0,b4=0,b5=0,b6=0;
    uint32_t c0=uint32_t(p2&0x7fffu),c1=uint32_t((p2>>15)&0x7fffu),c2=uint32_t((p2>>30)&0x7fffu),c3=0,c4=0,c5=0,c6=0;
    uint32_t d0=uint32_t(p3&0x7fffu),d1=uint32_t((p3>>15)&0x7fffu),d2=uint32_t((p3>>30)&0x7fffu),d3=0,d4=0,d5=0,d6=0;
    P10DCDirectGather64Word q0=0,q1=0,q2=0,q3=0;
#if P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64
    if(n0>3)q0=__ldg(D_P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64_RARE+uint32_t(p0>>48));
    if(n1>3)q1=__ldg(D_P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64_RARE+uint32_t(p1>>48));
    if(n2>3)q2=__ldg(D_P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64_RARE+uint32_t(p2>>48));
    if(n3>3)q3=__ldg(D_P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64_RARE+uint32_t(p3>>48));
#else
    if(n0>3)q0=__ldg(D_P10DC_RANKFORMULA_DIRECTGATHER64_RARE+uint32_t(p0>>48));
    if(n1>3)q1=__ldg(D_P10DC_RANKFORMULA_DIRECTGATHER64_RARE+uint32_t(p1>>48));
    if(n2>3)q2=__ldg(D_P10DC_RANKFORMULA_DIRECTGATHER64_RARE+uint32_t(p2>>48));
    if(n3>3)q3=__ldg(D_P10DC_RANKFORMULA_DIRECTGATHER64_RARE+uint32_t(p3>>48));
#endif
    if(n0>3){a3=uint32_t(q0&0x7fffu);a4=uint32_t((q0>>15)&0x7fffu);a5=uint32_t((q0>>30)&0x7fffu);a6=uint32_t((q0>>45)&0x7fffu);}
    if(n1>3){b3=uint32_t(q1&0x7fffu);b4=uint32_t((q1>>15)&0x7fffu);b5=uint32_t((q1>>30)&0x7fffu);b6=uint32_t((q1>>45)&0x7fffu);}
    if(n2>3){c3=uint32_t(q2&0x7fffu);c4=uint32_t((q2>>15)&0x7fffu);c5=uint32_t((q2>>30)&0x7fffu);c6=uint32_t((q2>>45)&0x7fffu);}
    if(n3>3){d3=uint32_t(q3&0x7fffu);d4=uint32_t((q3>>15)&0x7fffu);d5=uint32_t((q3>>30)&0x7fffu);d6=uint32_t((q3>>45)&0x7fffu);}

#if P10DC_RANKFORMULA_CPASYNC_PAIR
#define P10DC_QUAD_CPASYNC(slot,rank,count,need) \
    p10dc_rankformula_cpasync_u32( \
        p10dc_rankformula_cpasync_slot(slot), source_row + (rank), (count) > (need))
    P10DC_QUAD_CPASYNC(0,a0,n0,0); P10DC_QUAD_CPASYNC(1,a1,n0,1); P10DC_QUAD_CPASYNC(2,a2,n0,2); P10DC_QUAD_CPASYNC(3,a3,n0,3); P10DC_QUAD_CPASYNC(4,a4,n0,4); P10DC_QUAD_CPASYNC(5,a5,n0,5); P10DC_QUAD_CPASYNC(6,a6,n0,6); p10dc_rankformula_cpasync_commit();
    P10DC_QUAD_CPASYNC(7,b0,n1,0); P10DC_QUAD_CPASYNC(8,b1,n1,1); P10DC_QUAD_CPASYNC(9,b2,n1,2); P10DC_QUAD_CPASYNC(10,b3,n1,3); P10DC_QUAD_CPASYNC(11,b4,n1,4); P10DC_QUAD_CPASYNC(12,b5,n1,5); P10DC_QUAD_CPASYNC(13,b6,n1,6); p10dc_rankformula_cpasync_commit();
    P10DC_QUAD_CPASYNC(14,c0,n2,0); P10DC_QUAD_CPASYNC(15,c1,n2,1); P10DC_QUAD_CPASYNC(16,c2,n2,2); P10DC_QUAD_CPASYNC(17,c3,n2,3); P10DC_QUAD_CPASYNC(18,c4,n2,4); P10DC_QUAD_CPASYNC(19,c5,n2,5); P10DC_QUAD_CPASYNC(20,c6,n2,6); p10dc_rankformula_cpasync_commit();
    P10DC_QUAD_CPASYNC(21,d0,n3,0); P10DC_QUAD_CPASYNC(22,d1,n3,1); P10DC_QUAD_CPASYNC(23,d2,n3,2); P10DC_QUAD_CPASYNC(24,d3,n3,3); P10DC_QUAD_CPASYNC(25,d4,n3,4); P10DC_QUAD_CPASYNC(26,d5,n3,5); P10DC_QUAD_CPASYNC(27,d6,n3,6); p10dc_rankformula_cpasync_commit();
#undef P10DC_QUAD_CPASYNC
    p10dc_rankformula_cpasync_wait_all();
#define P10DC_QUAD_SLOT(i) BkczCrossAccum(*p10dc_rankformula_cpasync_slot(i))
    const BkczCrossAccum as03=p10dc_rankformula_accum_add(p10dc_rankformula_accum_add(P10DC_QUAD_SLOT(0),P10DC_QUAD_SLOT(1)),p10dc_rankformula_accum_add(P10DC_QUAD_SLOT(2),P10DC_QUAD_SLOT(3)));
    const BkczCrossAccum bs03=p10dc_rankformula_accum_add(p10dc_rankformula_accum_add(P10DC_QUAD_SLOT(7),P10DC_QUAD_SLOT(8)),p10dc_rankformula_accum_add(P10DC_QUAD_SLOT(9),P10DC_QUAD_SLOT(10)));
    const BkczCrossAccum cs03=p10dc_rankformula_accum_add(p10dc_rankformula_accum_add(P10DC_QUAD_SLOT(14),P10DC_QUAD_SLOT(15)),p10dc_rankformula_accum_add(P10DC_QUAD_SLOT(16),P10DC_QUAD_SLOT(17)));
    const BkczCrossAccum ds03=p10dc_rankformula_accum_add(p10dc_rankformula_accum_add(P10DC_QUAD_SLOT(21),P10DC_QUAD_SLOT(22)),p10dc_rankformula_accum_add(P10DC_QUAD_SLOT(23),P10DC_QUAD_SLOT(24)));
    const BkczCrossAccum as46=p10dc_rankformula_accum_add(p10dc_rankformula_accum_add(P10DC_QUAD_SLOT(4),P10DC_QUAD_SLOT(5)),P10DC_QUAD_SLOT(6));
    const BkczCrossAccum bs46=p10dc_rankformula_accum_add(p10dc_rankformula_accum_add(P10DC_QUAD_SLOT(11),P10DC_QUAD_SLOT(12)),P10DC_QUAD_SLOT(13));
    const BkczCrossAccum cs46=p10dc_rankformula_accum_add(p10dc_rankformula_accum_add(P10DC_QUAD_SLOT(18),P10DC_QUAD_SLOT(19)),P10DC_QUAD_SLOT(20));
    const BkczCrossAccum ds46=p10dc_rankformula_accum_add(p10dc_rankformula_accum_add(P10DC_QUAD_SLOT(25),P10DC_QUAD_SLOT(26)),P10DC_QUAD_SLOT(27));
#undef P10DC_QUAD_SLOT
    return P10DCRankFormulaQuadAccum{
        p10dc_rankformula_accum_add(as03,as46),p10dc_rankformula_accum_add(bs03,bs46),
        p10dc_rankformula_accum_add(cs03,cs46),p10dc_rankformula_accum_add(ds03,ds46)};
#else
    BkczCrossAccum av0=0,av1=0,av2=0,av3=0,bv0=0,bv1=0,bv2=0,bv3=0;
    BkczCrossAccum cv0=0,cv1=0,cv2=0,cv3=0,dv0=0,dv1=0,dv2=0,dv3=0;
    if(n0>0)av0=BkczCrossAccum(__ldg(source_row+a0)); if(n0>1)av1=BkczCrossAccum(__ldg(source_row+a1)); if(n0>2)av2=BkczCrossAccum(__ldg(source_row+a2)); if(n0>3)av3=BkczCrossAccum(__ldg(source_row+a3));
    if(n1>0)bv0=BkczCrossAccum(__ldg(source_row+b0)); if(n1>1)bv1=BkczCrossAccum(__ldg(source_row+b1)); if(n1>2)bv2=BkczCrossAccum(__ldg(source_row+b2)); if(n1>3)bv3=BkczCrossAccum(__ldg(source_row+b3));
    if(n2>0)cv0=BkczCrossAccum(__ldg(source_row+c0)); if(n2>1)cv1=BkczCrossAccum(__ldg(source_row+c1)); if(n2>2)cv2=BkczCrossAccum(__ldg(source_row+c2)); if(n2>3)cv3=BkczCrossAccum(__ldg(source_row+c3));
    if(n3>0)dv0=BkczCrossAccum(__ldg(source_row+d0)); if(n3>1)dv1=BkczCrossAccum(__ldg(source_row+d1)); if(n3>2)dv2=BkczCrossAccum(__ldg(source_row+d2)); if(n3>3)dv3=BkczCrossAccum(__ldg(source_row+d3));
    const BkczCrossAccum as03=p10dc_rankformula_accum_add(p10dc_rankformula_accum_add(av0,av1),p10dc_rankformula_accum_add(av2,av3));
    const BkczCrossAccum bs03=p10dc_rankformula_accum_add(p10dc_rankformula_accum_add(bv0,bv1),p10dc_rankformula_accum_add(bv2,bv3));
    const BkczCrossAccum cs03=p10dc_rankformula_accum_add(p10dc_rankformula_accum_add(cv0,cv1),p10dc_rankformula_accum_add(cv2,cv3));
    const BkczCrossAccum ds03=p10dc_rankformula_accum_add(p10dc_rankformula_accum_add(dv0,dv1),p10dc_rankformula_accum_add(dv2,dv3));

    BkczCrossAccum av4=0,av5=0,av6=0,bv4=0,bv5=0,bv6=0,cv4=0,cv5=0,cv6=0,dv4=0,dv5=0,dv6=0;
    if(n0>4)av4=BkczCrossAccum(__ldg(source_row+a4)); if(n0>5)av5=BkczCrossAccum(__ldg(source_row+a5)); if(n0>6)av6=BkczCrossAccum(__ldg(source_row+a6));
    if(n1>4)bv4=BkczCrossAccum(__ldg(source_row+b4)); if(n1>5)bv5=BkczCrossAccum(__ldg(source_row+b5)); if(n1>6)bv6=BkczCrossAccum(__ldg(source_row+b6));
    if(n2>4)cv4=BkczCrossAccum(__ldg(source_row+c4)); if(n2>5)cv5=BkczCrossAccum(__ldg(source_row+c5)); if(n2>6)cv6=BkczCrossAccum(__ldg(source_row+c6));
    if(n3>4)dv4=BkczCrossAccum(__ldg(source_row+d4)); if(n3>5)dv5=BkczCrossAccum(__ldg(source_row+d5)); if(n3>6)dv6=BkczCrossAccum(__ldg(source_row+d6));
    const BkczCrossAccum as46=p10dc_rankformula_accum_add(p10dc_rankformula_accum_add(av4,av5),av6);
    const BkczCrossAccum bs46=p10dc_rankformula_accum_add(p10dc_rankformula_accum_add(bv4,bv5),bv6);
    const BkczCrossAccum cs46=p10dc_rankformula_accum_add(p10dc_rankformula_accum_add(cv4,cv5),cv6);
    const BkczCrossAccum ds46=p10dc_rankformula_accum_add(p10dc_rankformula_accum_add(dv4,dv5),dv6);
    return P10DCRankFormulaQuadAccum{
        p10dc_rankformula_accum_add(as03,as46),p10dc_rankformula_accum_add(bs03,bs46),
        p10dc_rankformula_accum_add(cs03,cs46),p10dc_rankformula_accum_add(ds03,ds46)};
#endif
}
#endif
