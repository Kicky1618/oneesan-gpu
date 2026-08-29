#pragma once

#include "ramstream32_bucket_precomputed_high_ctx_compact.cuh"

// The flat chunk scheduler leaves the whole CTA idle while lane 0 expands a
// compact prectx into runtime pointers.  Expansion contains up to
// BKCZ_MAX_LOCAL independent affine row resolutions plus one CROSS resolution.
// Use the first warp as a tiny cooperative decoder: lanes [0,local_n) resolve
// local rows, lane BKCZ_MAX_LOCAL resolves CROSS, and lane 0 publishes local_n.
// The caller brackets this helper with warp/block synchronization.
static_assert(BKCZ_MAX_LOCAL + 1 <= 32,
              "warp-cooperative compact prectx requires local+CROSS lanes <= warp size");

__device__ __forceinline__ void p10dc_apply_forward_compact_prectx_warpcoop(
    P10DCDirectHighResolvedCtx& c, uint32_t qi, bool nn
) {
    const P10DCHighClosureCompactPreCtx* const z =
        (nn ? D_P10DC_COMPACT_PRECTX_FWD_NN : D_P10DC_COMPACT_PRECTX_FWD_NRNL) + qi;
    const uint32_t lane = uint32_t(threadIdx.x) & 31u;

    if (lane < uint32_t(BKCZ_MAX_LOCAL)) {
        const uint32_t n = uint32_t(z->local_n);
        if (lane < n) {
            const uint32_t ref = z->local_ref[lane];
            const uint32_t hs = uint32_t(z->fixed_hs);
            c.local_base[lane] = p10dc_high_row_ref_resolve_unchecked(ref, hs);
        }
        if (lane == 0u) c.local_n = uint8_t(n);
    } else if (lane == uint32_t(BKCZ_MAX_LOCAL)) {
        const uint32_t depth = uint32_t(z->cross_depth);
        const uint32_t hs = uint32_t(z->fixed_hs);
        c.cross_depth = depth;
        c.cross_base = depth
            ? p10dc_high_row_ref_resolve_unchecked(z->cross_ref, hs + 2u)
            : nullptr;
    }
}

__device__ __forceinline__ void p10dc_apply_reverse_compact_prectx_warpcoop(
    P10DCDirectHighResolvedCtx& c, uint32_t qi, uint32_t kind
) {
    const P10DCHighClosureCompactPreCtx* z = nullptr;
    if (kind == CPU_ORBIT_NN) z = D_P10DC_COMPACT_PRECTX_REV_NN + qi;
    else if (kind == CPU_ORBIT_NR) z = D_P10DC_COMPACT_PRECTX_REV_NR + qi;
    else z = D_P10DC_COMPACT_PRECTX_REV_NL + qi;

    const uint32_t lane = uint32_t(threadIdx.x) & 31u;
    if (lane < uint32_t(BKCZ_MAX_LOCAL)) {
        const uint32_t n = uint32_t(z->local_n);
        if (lane < n) {
            const uint32_t ref = z->local_ref[lane];
            const uint32_t hs = uint32_t(z->fixed_hs);
            c.local_base[lane] = p10dc_high_row_ref_resolve_unchecked(ref, hs);
        }
        if (lane == 0u) c.local_n = uint8_t(n);
    } else if (lane == uint32_t(BKCZ_MAX_LOCAL)) {
        const uint32_t depth = uint32_t(z->cross_depth);
        const uint32_t hs = uint32_t(z->fixed_hs);
        c.cross_depth = depth;
        c.cross_base = depth
            ? p10dc_high_row_ref_resolve_unchecked(z->cross_ref, hs + 2u)
            : nullptr;
    }
}
