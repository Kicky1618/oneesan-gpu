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

__device__ __forceinline__ void p10dc_apply_loaded_compact_prectx(
    P10DCDirectHighResolvedCtx& c,
    const P10DCHighClosureCompactPreCtx& z
) {
    c.local_n = z.local_n;
#pragma unroll
    for (uint32_t i = 0; i < BKCZ_MAX_LOCAL; ++i)
        if (i < uint32_t(z.local_n))
            c.local_base[i] = p10dc_high_row_ref_resolve_unchecked(z.local_ref[i], z.fixed_hs);
    c.cross_depth = uint32_t(z.cross_depth);
    c.cross_base = z.cross_depth
        ? p10dc_high_row_ref_resolve_unchecked(z.cross_ref, uint32_t(z.fixed_hs) + 2u)
        : nullptr;
}

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

__device__ __forceinline__ P10DCHighClosureCompactPreCtx
p10dc_load_forward_compact_prectx_flat(
    uint32_t q, bool nn
) {
    return nn ? D_P10DC_COMPACT_PRECTX_FWD_NN[q]
              : D_P10DC_COMPACT_PRECTX_FWD_NRNL[q];
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

__device__ __forceinline__ P10DCHighClosureCompactPreCtx
p10dc_load_reverse_compact_prectx_flat(
    uint32_t q, uint32_t kind
) {
    if (kind == CPU_ORBIT_NN) return D_P10DC_COMPACT_PRECTX_REV_NN[q];
    if (kind == CPU_ORBIT_NR) return D_P10DC_COMPACT_PRECTX_REV_NR[q];
    return D_P10DC_COMPACT_PRECTX_REV_NL[q];
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
                  << " fused_context_load=1"
                  << " forward=" << P10DC_RANKFORMULA_PRECTX_FORWARD
                  << " reverse=" << P10DC_RANKFORMULA_PRECTX_REVERSE
                  << '\n';
    }
};
