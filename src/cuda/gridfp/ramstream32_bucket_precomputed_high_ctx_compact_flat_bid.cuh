#pragma once

#include "ramstream32_bucket_precomputed_high_ctx_compact.cuh"

// The compact prectx already has one unused byte for alignment. Flat orbit
// scheduling needs the source main-block id before it can build xb/jb/db, so
// cache that id in the existing pad byte. W28 has only 45 main blocks. This
// adds no resident bytes and leaves the ordinary orbit-CTA ABI unchanged.
static_assert(P10DC_HIGH_ROW_AFFINE_BLOCKS <= 255u,
              "compact flat-bid cache stores main bid in one byte");
static_assert(sizeof(P10DCHighClosureCompactPreCtx) ==
                  sizeof(uint32_t) * (BKCZ_MAX_LOCAL + 2u),
              "flat-bid cache must not grow compact prectx");

#if P10DC_RANKFORMULA_PRECTX_FORWARD
__global__ void p10dc_annotate_forward_compact_prectx_bid_kernel(
    P10DCHighClosureCompactPreCtx* nn,
    P10DCHighClosureCompactPreCtx* nrnl
) {
    const uint32_t nb = D_BKF_MAIN_NBLOCKS;
    if (!nb) return;
    const uint32_t flat = uint32_t(blockIdx.x);
    const uint32_t pi = flat / nb;
    const uint32_t bid = flat - pi * nb;
    if (pi >= uint32_t(HIGH_LUT_K) || bid >= nb || bid > 255u) return;
    const uint32_t oi = uint32_t(size_t(pi) * D_BKF_HIGH_PITCH + bid);
    for (uint32_t q = D_BKF_HIGH_NN_OFF[oi] + threadIdx.x;
         q < D_BKF_HIGH_NN_OFF[oi + 1u]; q += blockDim.x)
        nn[q].pad = uint8_t(bid);
    for (uint32_t q = D_BKF_HIGH_NRNL_OFF[oi] + threadIdx.x;
         q < D_BKF_HIGH_NRNL_OFF[oi + 1u]; q += blockDim.x)
        nrnl[q].pad = uint8_t(bid);
}

__device__ __forceinline__ uint32_t p10dc_forward_compact_prectx_flat_bid(
    uint32_t q, bool nn
) {
    return uint32_t(nn ? D_P10DC_COMPACT_PRECTX_FWD_NN[q].pad
                       : D_P10DC_COMPACT_PRECTX_FWD_NRNL[q].pad);
}
#endif

#if P10DC_RANKFORMULA_PRECTX_REVERSE
__global__ void p10dc_annotate_reverse_compact_prectx_bid_kernel(
    P10DCHighClosureCompactPreCtx* nn,
    P10DCHighClosureCompactPreCtx* nr,
    P10DCHighClosureCompactPreCtx* nl
) {
    const uint32_t nb = D_BKF_MAIN_NBLOCKS;
    if (!nb) return;
    const uint32_t flat = uint32_t(blockIdx.x);
    const uint32_t pi = flat / nb;
    const uint32_t bid = flat - pi * nb;
    if (pi >= uint32_t(HIGH_LUT_K) || bid >= nb || bid > 255u) return;
    const uint32_t oi = uint32_t(size_t(pi) * D_RS54_PITCH + bid);
    for (uint32_t q = D_RS54_HIGH_NN_OFF[oi] + threadIdx.x;
         q < D_RS54_HIGH_NN_OFF[oi + 1u]; q += blockDim.x)
        nn[q].pad = uint8_t(bid);
    for (uint32_t q = D_RS54_HIGH_NR_OFF[oi] + threadIdx.x;
         q < D_RS54_HIGH_NR_OFF[oi + 1u]; q += blockDim.x)
        nr[q].pad = uint8_t(bid);
    for (uint32_t q = D_RS54_HIGH_NL_OFF[oi] + threadIdx.x;
         q < D_RS54_HIGH_NL_OFF[oi + 1u]; q += blockDim.x)
        nl[q].pad = uint8_t(bid);
}

__device__ __forceinline__ uint32_t p10dc_reverse_compact_prectx_flat_bid(
    uint32_t q, uint32_t kind
) {
    if (kind == CPU_ORBIT_NN) return uint32_t(D_P10DC_COMPACT_PRECTX_REV_NN[q].pad);
    if (kind == CPU_ORBIT_NR) return uint32_t(D_P10DC_COMPACT_PRECTX_REV_NR[q].pad);
    return uint32_t(D_P10DC_COMPACT_PRECTX_REV_NL[q].pad);
}
#endif

template<class BaseTables>
struct BucketFusedCompactPrecomputedHighCtxFlatBidTables
    : BucketFusedCompactPrecomputedHighCtxTables<BaseTables> {
    using Parent = BucketFusedCompactPrecomputedHighCtxTables<BaseTables>;

    void bind_owner(
        uint32_t fixed,
        const BucketPhysicalLayoutHost& buckets,
        const std::array<Count*, BUCKET_NGPU>& slot
    ) {
        Parent::bind_owner(fixed, buckets, slot);
        const uint32_t contexts = uint32_t(HIGH_LUT_K) * this->prectx_main_blocks;
        if (contexts) {
#if P10DC_RANKFORMULA_PRECTX_FORWARD
            p10dc_annotate_forward_compact_prectx_bid_kernel<<<contexts, 128>>>(
                this->prectx_fwd_nn, this->prectx_fwd_nrnl);
            ck(cudaGetLastError(), "p10dc compact flat bid forward annotate");
#endif
#if P10DC_RANKFORMULA_PRECTX_REVERSE
            p10dc_annotate_reverse_compact_prectx_bid_kernel<<<contexts, 128>>>(
                this->prectx_rev_nn, this->prectx_rev_nr, this->prectx_rev_nl);
            ck(cudaGetLastError(), "p10dc compact flat bid reverse annotate");
#endif
            ck(cudaDeviceSynchronize(), "p10dc compact flat bid annotate sync");
        }
        std::cerr << "p10dc_compact_flat_bid fixed_owner=" << fixed
                  << " bytes_added=0"
                  << " bid_storage=existing_pad_u8"
                  << " forward=" << P10DC_RANKFORMULA_PRECTX_FORWARD
                  << " reverse=" << P10DC_RANKFORMULA_PRECTX_REVERSE
                  << '\n';
    }
};
