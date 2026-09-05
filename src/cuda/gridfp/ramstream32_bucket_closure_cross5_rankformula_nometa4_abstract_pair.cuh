#pragma once

#include "ramstream32_bucket_closure_cross5_rankformula_nometa4_abstract_mlp.cuh"

#ifndef P10DC_RANKFORMULA_PAIR_MLP
#define P10DC_RANKFORMULA_PAIR_MLP 0
#endif
#ifndef P10DC_RANKFORMULA_CPASYNC_PAIR
#define P10DC_RANKFORMULA_CPASYNC_PAIR 0
#endif
#ifndef P10DC_RANKFORMULA_DIRECTGATHER64
#define P10DC_RANKFORMULA_DIRECTGATHER64 0
#endif
#if P10DC_RANKFORMULA_DIRECTGATHER64
#include "ramstream32_bucket_closure_cross5_rankformula_nometa4_directgather64.cuh"
#endif
static_assert(P10DC_RANKFORMULA_PAIR_MLP == 0 || P10DC_RANKFORMULA_PAIR_MLP == 1,
              "P10DC_RANKFORMULA_PAIR_MLP must be 0 or 1");
static_assert(P10DC_RANKFORMULA_CPASYNC_PAIR == 0 || P10DC_RANKFORMULA_CPASYNC_PAIR == 1,
              "P10DC_RANKFORMULA_CPASYNC_PAIR must be 0 or 1");
#if P10DC_RANKFORMULA_PAIR_MLP
static_assert(P10DC_RANKFORMULA_DIRECTGATHER,
              "pair MLP currently requires DIRECTGATHER");
static_assert(P10DC_RANKFORMULA_DIRECTGATHER_DEPTHMAJOR,
              "pair MLP currently requires depth-major DIRECTGATHER");
static_assert(P10DC_RANKFORMULA_MLP_WINDOW4,
              "pair MLP requires WINDOW4 to bound register pressure");
#endif
#if P10DC_RANKFORMULA_CPASYNC_PAIR
static_assert(P10DC_RANKFORMULA_PAIR_MLP,
              "cp.async pair path requires PAIR_MLP");
static_assert(sizeof(Count) == 4,
              "cp.async pair path assumes 32-bit Count values");
extern __shared__ unsigned char p10dc_rankformula_dynamic_smem[];
#endif

struct P10DCRankFormulaPairAccum {
    BkczCrossAccum a;
    BkczCrossAccum b;
};

__device__ __forceinline__ uint4 p10dc_rankformula_directgather_depth_desc(
    uint32_t h, uint32_t rank, uint32_t depth
) {
    const size_t gi = size_t(D_P10DC_RANKFORMULA_DIRECTGATHER_DEPTH_OFF[
        h * P10DC_RANKFORMULA_ABSTRACT_SELECT_DEPTHS + (depth - 1u)]) + size_t(rank);
    return __ldg(D_P10DC_RANKFORMULA_DIRECTGATHER4 + gi);
}

#if P10DC_RANKFORMULA_CPASYNC_PAIR
__device__ __forceinline__ Count* p10dc_rankformula_cpasync_slot(uint32_t slot) {
    const size_t off = p10dc_direct_pair_scratch_offset_bytes(int(blockDim.x));
    Count* base = reinterpret_cast<Count*>(p10dc_rankformula_dynamic_smem + off);
    // Slot-major layout: for a fixed source ordinal, adjacent warp lanes target
    // adjacent shared words.  This avoids the 14-word per-thread stride and the
    // associated bank conflicts while keeping each thread's fourteen values private.
    return base + size_t(slot) * blockDim.x + threadIdx.x;
}

__device__ __forceinline__ void p10dc_rankformula_cpasync_u32(
    Count* dst, const Count* src, bool valid
) {
    if (!valid) {
        *dst = Count(0);
        return;
    }
#if __CUDA_ARCH__ >= 800
    const uint32_t sdst = uint32_t(__cvta_generic_to_shared(dst));
    const unsigned long long gsrc = reinterpret_cast<unsigned long long>(src);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(sdst), "l"(gsrc));
#else
    *dst = *src;
#endif
}

__device__ __forceinline__ void p10dc_rankformula_cpasync_commit() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;");
#endif
}

__device__ __forceinline__ void p10dc_rankformula_cpasync_wait_all() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;");
#endif
}
#endif

__device__ __forceinline__ P10DCRankFormulaPairAccum
p10dc_resolved_low_preimages_cross5_rankformula_nometa4_abstract_pair_fixed(
    uint32_t h, uint32_t rank0, uint32_t rank1, uint32_t depth,
    const Count* source_row
) {
#if !P10DC_RANKFORMULA_PAIR_MLP
    return P10DCRankFormulaPairAccum{
        p10dc_resolved_low_preimages_cross5_rankformula_nometa4_abstract_mlp_fixed(
            h, rank0, depth, source_row),
        p10dc_resolved_low_preimages_cross5_rankformula_nometa4_abstract_mlp_fixed(
            h, rank1, depth, source_row)};
#else
    if (!depth || depth > P10DC_RANKFORMULA_ABSTRACT_SELECT_DEPTHS)
        return P10DCRankFormulaPairAccum{0, 0};

    uint32_t n0 = 0, n1 = 0;
    uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0;
    uint32_t b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0, b6 = 0;
#if P10DC_RANKFORMULA_DIRECTGATHER64
    // Issue both compact 8-byte primary descriptors before decoding either
    // source list.  Only ranks with >3 sources issue an 8-byte rare descriptor.
    // This keeps the common descriptor traffic at 16 bytes for the pair while
    // retaining the two-column MLP schedule below.
    const size_t gi0 = p10dc_rankformula_directgather_index(h, rank0, depth);
    const size_t gi1 = p10dc_rankformula_directgather_index(h, rank1, depth);
    const P10DCDirectGather64Word p0 = __ldg(D_P10DC_RANKFORMULA_DIRECTGATHER64 + gi0);
    const P10DCDirectGather64Word p1 = __ldg(D_P10DC_RANKFORMULA_DIRECTGATHER64 + gi1);
    n0 = uint32_t((p0 >> 45) & 7u);
    n1 = uint32_t((p1 >> 45) & 7u);
    if (!(n0 | n1)) return P10DCRankFormulaPairAccum{0, 0};

    a0 = uint32_t(p0 & 0x7fffu);
    a1 = uint32_t((p0 >> 15) & 0x7fffu);
    a2 = uint32_t((p0 >> 30) & 0x7fffu);
    b0 = uint32_t(p1 & 0x7fffu);
    b1 = uint32_t((p1 >> 15) & 0x7fffu);
    b2 = uint32_t((p1 >> 30) & 0x7fffu);

    P10DCDirectGather64Word q0 = 0, q1 = 0;
    if (n0 > 3)
        q0 = __ldg(D_P10DC_RANKFORMULA_DIRECTGATHER64_RARE + uint32_t(p0 >> 48));
    if (n1 > 3)
        q1 = __ldg(D_P10DC_RANKFORMULA_DIRECTGATHER64_RARE + uint32_t(p1 >> 48));
    if (n0 > 3) {
        a3 = uint32_t(q0 & 0x7fffu);
        a4 = uint32_t((q0 >> 15) & 0x7fffu);
        a5 = uint32_t((q0 >> 30) & 0x7fffu);
        a6 = uint32_t((q0 >> 45) & 0x7fffu);
    }
    if (n1 > 3) {
        b3 = uint32_t(q1 & 0x7fffu);
        b4 = uint32_t((q1 >> 15) & 0x7fffu);
        b5 = uint32_t((q1 >> 30) & 0x7fffu);
        b6 = uint32_t((q1 >> 45) & 0x7fffu);
    }
#else
    // Issue both 16-byte descriptors before consuming either one.  With the
    // depth-major table, every warp still sees contiguous descriptor accesses,
    // while each lane now has two independent descriptor requests in flight.
    const uint4 d0 = p10dc_rankformula_directgather_depth_desc(h, rank0, depth);
    const uint4 d1 = p10dc_rankformula_directgather_depth_desc(h, rank1, depth);
    n0 = d0.w >> 16;
    n1 = d1.w >> 16;
    if (!(n0 | n1)) return P10DCRankFormulaPairAccum{0, 0};

    a0 = d0.x & 0xffffu; a1 = d0.x >> 16;
    a2 = d0.y & 0xffffu; a3 = d0.y >> 16;
    a4 = d0.z & 0xffffu; a5 = d0.z >> 16;
    a6 = d0.w & 0xffffu;
    b0 = d1.x & 0xffffu; b1 = d1.x >> 16;
    b2 = d1.y & 0xffffu; b3 = d1.y >> 16;
    b4 = d1.z & 0xffffu; b5 = d1.z >> 16;
    b6 = d1.w & 0xffffu;
#endif

#if P10DC_RANKFORMULA_CPASYNC_PAIR
    // Stage all selected CROSS sources through shared memory.  Descriptor
    // decoding above is independent of 16-byte versus compressed 8-byte mode,
    // so the same fourteen source slots serve both encodings.
    p10dc_rankformula_cpasync_u32(p10dc_rankformula_cpasync_slot(0), source_row + a0, n0 > 0);
    p10dc_rankformula_cpasync_u32(p10dc_rankformula_cpasync_slot(1), source_row + a1, n0 > 1);
    p10dc_rankformula_cpasync_u32(p10dc_rankformula_cpasync_slot(2), source_row + a2, n0 > 2);
    p10dc_rankformula_cpasync_u32(p10dc_rankformula_cpasync_slot(3), source_row + a3, n0 > 3);
    p10dc_rankformula_cpasync_u32(p10dc_rankformula_cpasync_slot(4), source_row + a4, n0 > 4);
    p10dc_rankformula_cpasync_u32(p10dc_rankformula_cpasync_slot(5), source_row + a5, n0 > 5);
    p10dc_rankformula_cpasync_u32(p10dc_rankformula_cpasync_slot(6), source_row + a6, n0 > 6);
    p10dc_rankformula_cpasync_commit();

    p10dc_rankformula_cpasync_u32(p10dc_rankformula_cpasync_slot(7), source_row + b0, n1 > 0);
    p10dc_rankformula_cpasync_u32(p10dc_rankformula_cpasync_slot(8), source_row + b1, n1 > 1);
    p10dc_rankformula_cpasync_u32(p10dc_rankformula_cpasync_slot(9), source_row + b2, n1 > 2);
    p10dc_rankformula_cpasync_u32(p10dc_rankformula_cpasync_slot(10), source_row + b3, n1 > 3);
    p10dc_rankformula_cpasync_u32(p10dc_rankformula_cpasync_slot(11), source_row + b4, n1 > 4);
    p10dc_rankformula_cpasync_u32(p10dc_rankformula_cpasync_slot(12), source_row + b5, n1 > 5);
    p10dc_rankformula_cpasync_u32(p10dc_rankformula_cpasync_slot(13), source_row + b6, n1 > 6);
    p10dc_rankformula_cpasync_commit();
    p10dc_rankformula_cpasync_wait_all();

    const BkczCrossAccum a03 = p10dc_rankformula_accum_add(
        p10dc_rankformula_accum_add(
            BkczCrossAccum(*p10dc_rankformula_cpasync_slot(0)),
            BkczCrossAccum(*p10dc_rankformula_cpasync_slot(1))),
        p10dc_rankformula_accum_add(
            BkczCrossAccum(*p10dc_rankformula_cpasync_slot(2)),
            BkczCrossAccum(*p10dc_rankformula_cpasync_slot(3))));
    const BkczCrossAccum a46 = p10dc_rankformula_accum_add(
        p10dc_rankformula_accum_add(
            BkczCrossAccum(*p10dc_rankformula_cpasync_slot(4)),
            BkczCrossAccum(*p10dc_rankformula_cpasync_slot(5))),
        BkczCrossAccum(*p10dc_rankformula_cpasync_slot(6)));
    const BkczCrossAccum b03 = p10dc_rankformula_accum_add(
        p10dc_rankformula_accum_add(
            BkczCrossAccum(*p10dc_rankformula_cpasync_slot(7)),
            BkczCrossAccum(*p10dc_rankformula_cpasync_slot(8))),
        p10dc_rankformula_accum_add(
            BkczCrossAccum(*p10dc_rankformula_cpasync_slot(9)),
            BkczCrossAccum(*p10dc_rankformula_cpasync_slot(10))));
    const BkczCrossAccum b46 = p10dc_rankformula_accum_add(
        p10dc_rankformula_accum_add(
            BkczCrossAccum(*p10dc_rankformula_cpasync_slot(11)),
            BkczCrossAccum(*p10dc_rankformula_cpasync_slot(12))),
        BkczCrossAccum(*p10dc_rankformula_cpasync_slot(13)));
    return P10DCRankFormulaPairAccum{
        p10dc_rankformula_accum_add(a03, a46),
        p10dc_rankformula_accum_add(b03, b46)};
#else
    // Four-wide window for both columns.  The loads for column B are issued
    // before column A is reduced, increasing outstanding requests without
    // keeping fourteen source values live simultaneously.
    BkczCrossAccum a03_0 = 0, a03_1 = 0, a03_2 = 0, a03_3 = 0;
    BkczCrossAccum b03_0 = 0, b03_1 = 0, b03_2 = 0, b03_3 = 0;
    if (n0 > 0) a03_0 = BkczCrossAccum(__ldg(source_row + a0));
    if (n0 > 1) a03_1 = BkczCrossAccum(__ldg(source_row + a1));
    if (n0 > 2) a03_2 = BkczCrossAccum(__ldg(source_row + a2));
    if (n0 > 3) a03_3 = BkczCrossAccum(__ldg(source_row + a3));
    if (n1 > 0) b03_0 = BkczCrossAccum(__ldg(source_row + b0));
    if (n1 > 1) b03_1 = BkczCrossAccum(__ldg(source_row + b1));
    if (n1 > 2) b03_2 = BkczCrossAccum(__ldg(source_row + b2));
    if (n1 > 3) b03_3 = BkczCrossAccum(__ldg(source_row + b3));
    const BkczCrossAccum sa03 = p10dc_rankformula_accum_add(
        p10dc_rankformula_accum_add(a03_0, a03_1),
        p10dc_rankformula_accum_add(a03_2, a03_3));
    const BkczCrossAccum sb03 = p10dc_rankformula_accum_add(
        p10dc_rankformula_accum_add(b03_0, b03_1),
        p10dc_rankformula_accum_add(b03_2, b03_3));

    BkczCrossAccum a46_4 = 0, a46_5 = 0, a46_6 = 0;
    BkczCrossAccum b46_4 = 0, b46_5 = 0, b46_6 = 0;
    if (n0 > 4) a46_4 = BkczCrossAccum(__ldg(source_row + a4));
    if (n0 > 5) a46_5 = BkczCrossAccum(__ldg(source_row + a5));
    if (n0 > 6) a46_6 = BkczCrossAccum(__ldg(source_row + a6));
    if (n1 > 4) b46_4 = BkczCrossAccum(__ldg(source_row + b4));
    if (n1 > 5) b46_5 = BkczCrossAccum(__ldg(source_row + b5));
    if (n1 > 6) b46_6 = BkczCrossAccum(__ldg(source_row + b6));
    const BkczCrossAccum sa46 = p10dc_rankformula_accum_add(
        p10dc_rankformula_accum_add(a46_4, a46_5), a46_6);
    const BkczCrossAccum sb46 = p10dc_rankformula_accum_add(
        p10dc_rankformula_accum_add(b46_4, b46_5), b46_6);

    return P10DCRankFormulaPairAccum{
        p10dc_rankformula_accum_add(sa03, sa46),
        p10dc_rankformula_accum_add(sb03, sb46)};
#endif
#endif
}
