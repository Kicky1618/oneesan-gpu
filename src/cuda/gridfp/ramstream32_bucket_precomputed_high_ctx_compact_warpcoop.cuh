#pragma once

#include "ramstream32_bucket_precomputed_high_ctx_compact.cuh"

// The flat schedulers leave worker lanes idle while one lane expands compact
// prectx into runtime pointers. Expansion contains up to BKCZ_MAX_LOCAL
// independent affine row resolutions plus one CROSS resolution. Use warp 0 as
// a cooperative decoder. The compact descriptor is consumed as one contiguous
// 40-byte footprint for K14:
//   lanes 0..7 : local_ref[0..7] (32B)
//   lane 8     : cross_ref        (4B)
//   lane 0     : packed byte tail (4B: local_n,cross_depth,fixed_hs,pad)
// The tail can be loaded before orbit setup and reused by the decoder; pad is
// also the cached flat bid when PRECTX_FLAT_BID is installed.
static_assert(BKCZ_MAX_LOCAL + 1 <= 32,
              "warp-cooperative compact prectx requires local+CROSS lanes <= warp size");
static_assert(offsetof(P10DCHighClosureCompactPreCtx, local_n) % alignof(uint32_t) == 0,
              "compact prectx byte tail must stay uint32 aligned");
static_assert(offsetof(P10DCHighClosureCompactPreCtx, cross_ref) ==
                  sizeof(uint32_t) * BKCZ_MAX_LOCAL,
              "compact prectx CROSS ref must follow local refs contiguously");

__device__ __forceinline__ uint32_t p10dc_compact_prectx_warpcoop_load_meta(
    const P10DCHighClosureCompactPreCtx* z
) {
    const uint32_t lane = uint32_t(threadIdx.x) & 31u;
    const unsigned active = __activemask();
    uint32_t meta = 0;
    if (lane == 0u) {
        const auto* tail = reinterpret_cast<const uint32_t*>(&z->local_n);
        meta = __ldg(tail);
    }
    return __shfl_sync(active, meta, 0);
}

__device__ __forceinline__ void p10dc_apply_compact_prectx_warpcoop_ptr_meta(
    P10DCDirectHighResolvedCtx& c,
    const P10DCHighClosureCompactPreCtx* z,
    uint32_t meta
) {
    const uint32_t lane = uint32_t(threadIdx.x) & 31u;
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

__device__ __forceinline__ void p10dc_apply_compact_prectx_warpcoop_ptr(
    P10DCDirectHighResolvedCtx& c,
    const P10DCHighClosureCompactPreCtx* z
) {
    const uint32_t meta = p10dc_compact_prectx_warpcoop_load_meta(z);
    p10dc_apply_compact_prectx_warpcoop_ptr_meta(c, z, meta);
}

__device__ __forceinline__ const P10DCHighClosureCompactPreCtx*
p10dc_forward_compact_prectx_warpcoop_ptr(uint32_t qi, bool nn) {
    return (nn ? D_P10DC_COMPACT_PRECTX_FWD_NN : D_P10DC_COMPACT_PRECTX_FWD_NRNL) + qi;
}

__device__ __forceinline__ const P10DCHighClosureCompactPreCtx*
p10dc_reverse_compact_prectx_warpcoop_ptr(uint32_t qi, uint32_t kind) {
    if (kind == CPU_ORBIT_NN) return D_P10DC_COMPACT_PRECTX_REV_NN + qi;
    if (kind == CPU_ORBIT_NR) return D_P10DC_COMPACT_PRECTX_REV_NR + qi;
    return D_P10DC_COMPACT_PRECTX_REV_NL + qi;
}

__device__ __forceinline__ void p10dc_apply_forward_compact_prectx_warpcoop(
    P10DCDirectHighResolvedCtx& c, uint32_t qi, bool nn
) {
    p10dc_apply_compact_prectx_warpcoop_ptr(
        c, p10dc_forward_compact_prectx_warpcoop_ptr(qi, nn));
}

__device__ __forceinline__ void p10dc_apply_reverse_compact_prectx_warpcoop(
    P10DCDirectHighResolvedCtx& c, uint32_t qi, uint32_t kind
) {
    p10dc_apply_compact_prectx_warpcoop_ptr(
        c, p10dc_reverse_compact_prectx_warpcoop_ptr(qi, kind));
}
