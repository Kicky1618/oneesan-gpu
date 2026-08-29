#pragma once

#include "ramstream32_bucket_precomputed_high_ctx_compact.cuh"

// The flat chunk scheduler leaves the whole CTA idle while lane 0 expands a
// compact prectx into runtime pointers. Expansion contains up to
// BKCZ_MAX_LOCAL independent affine row resolutions plus one CROSS resolution.
// Use the first warp as a tiny cooperative decoder. The compact descriptor is
// deliberately consumed as one contiguous 40-byte footprint for K14:
//   lanes 0..7 : local_ref[0..7] (32B)
//   lane 8     : cross_ref        (4B)
//   lane 0     : packed byte tail (4B: local_n,cross_depth,fixed_hs,pad)
// The tail is broadcast once with a warp shuffle instead of reloading local_n
// and fixed_hs independently in every participating lane.
static_assert(BKCZ_MAX_LOCAL + 1 <= 32,
              "warp-cooperative compact prectx requires local+CROSS lanes <= warp size");
static_assert(offsetof(P10DCHighClosureCompactPreCtx, local_n) % alignof(uint32_t) == 0,
              "compact prectx byte tail must stay uint32 aligned");
static_assert(offsetof(P10DCHighClosureCompactPreCtx, cross_ref) ==
                  sizeof(uint32_t) * BKCZ_MAX_LOCAL,
              "compact prectx CROSS ref must follow local refs contiguously");

__device__ __forceinline__ void p10dc_apply_compact_prectx_warpcoop_ptr(
    P10DCDirectHighResolvedCtx& c,
    const P10DCHighClosureCompactPreCtx* z
) {
    const uint32_t lane = uint32_t(threadIdx.x) & 31u;
    const unsigned active = __activemask();

    uint32_t meta = 0;
    if (lane == 0u) {
        const auto* tail = reinterpret_cast<const uint32_t*>(&z->local_n);
        meta = __ldg(tail);
    }
    meta = __shfl_sync(active, meta, 0);
    const uint32_t n = meta & 0xffu;
    const uint32_t depth = (meta >> 8) & 0xffu;
    const uint32_t hs = (meta >> 16) & 0xffu;

    if (lane < uint32_t(BKCZ_MAX_LOCAL)) {
        if (lane < n) {
            const uint32_t ref = __ldg(&z->local_ref[lane]);
            c.local_base[lane] = p10dc_high_row_ref_resolve_unchecked(ref, hs);
        }
        if (lane == 0u) c.local_n = uint8_t(n);
    } else if (lane == uint32_t(BKCZ_MAX_LOCAL)) {
        c.cross_depth = depth;
        if (depth) {
            const uint32_t ref = __ldg(&z->cross_ref);
            c.cross_base = p10dc_high_row_ref_resolve_unchecked(ref, hs + 2u);
        } else {
            c.cross_base = nullptr;
        }
    }
}

__device__ __forceinline__ void p10dc_apply_forward_compact_prectx_warpcoop(
    P10DCDirectHighResolvedCtx& c, uint32_t qi, bool nn
) {
    const P10DCHighClosureCompactPreCtx* const z =
        (nn ? D_P10DC_COMPACT_PRECTX_FWD_NN : D_P10DC_COMPACT_PRECTX_FWD_NRNL) + qi;
    p10dc_apply_compact_prectx_warpcoop_ptr(c, z);
}

__device__ __forceinline__ void p10dc_apply_reverse_compact_prectx_warpcoop(
    P10DCDirectHighResolvedCtx& c, uint32_t qi, uint32_t kind
) {
    const P10DCHighClosureCompactPreCtx* z = nullptr;
    if (kind == CPU_ORBIT_NN) z = D_P10DC_COMPACT_PRECTX_REV_NN + qi;
    else if (kind == CPU_ORBIT_NR) z = D_P10DC_COMPACT_PRECTX_REV_NR + qi;
    else z = D_P10DC_COMPACT_PRECTX_REV_NL + qi;
    p10dc_apply_compact_prectx_warpcoop_ptr(c, z);
}
