#pragma once

#include "ramstream32_bucket_precomputed_high_ctx_compact_warpcoop.cuh"

#ifndef P10DC_ORBITCTA_CTX
#error "producer prectx warpcoop requires P10DC_ORBITCTA_CTX"
#endif
#ifndef P10DC_RANKFORMULA_PRECTX_FLAT_BID_FUSED
#define P10DC_RANKFORMULA_PRECTX_FLAT_BID_FUSED 0
#endif
static_assert(P10DC_RANKFORMULA_PRECTX_COMPACT == 1 &&
              P10DC_RANKFORMULA_PRECTX_FORWARD == 1 &&
              P10DC_RANKFORMULA_PRECTX_REVERSE == 1,
              "producer prectx warpcoop requires compact forward+reverse prectx");
static_assert(P10DC_RANKFORMULA_PRECTX_FLAT_BID_FUSED == 0,
              "producer prectx warpcoop uses distributed descriptor loads; fused lane0 load is a separate experiment");

// Called by warp 0 only. Lane 0 maps the queue item to q/kind and resolves the
// three direct I/O rows. Lanes 0..8 expand the compact closure row refs while
// worker warps consume the previous context. With PRECTX_FLAT_BID, the same
// 32-bit tail load provides all four bytes at once:
//   local_n | cross_depth<<8 | fixed_hs<<16 | cached_bid<<24.
// The cached bid therefore costs no extra descriptor load and the loaded meta is
// reused by the row-ref decoder instead of reloading the tail after orbit setup.
// stream_base is CTA-persistent shared metadata filled once at kernel entry;
// only lane 0 dereferences it, eliminating one HIGH stream-offset load per orbit
// without adding a producer-warp synchronization point.
__device__ __forceinline__ void
p10dc_orbitcta_flat_dynamic_pipe2_prepare_forward_producer_prectx_warpcoop(
    P10DC_ORBITCTA_CTX& c,
    uint32_t k_lane0,
    uint32_t base_off,
    const uint32_t* stream_base,
    uint32_t nblocks,
    int p
) {
    const uint32_t lane = uint32_t(threadIdx.x) & 31u;
    const unsigned active = __activemask();
    const uint32_t k = __shfl_sync(active, k_lane0, 0);
    if (k == 0xffffffffu) return;

    uint32_t q_lane0 = 0u;
    uint32_t nn_lane0 = 0u;
#if !P10DC_RANKFORMULA_PRECTX_FLAT_BID
    uint32_t bid_lane0 = 0xffffffffu;
#endif
    if (lane == 0u) {
        c.valid = 0;
        const bool nn = k < c.n0;
        const uint32_t q = nn
            ? stream_base[0] + k
            : stream_base[1] + k - c.n0;
        q_lane0 = q;
        nn_lane0 = uint32_t(nn);
#if !P10DC_RANKFORMULA_PRECTX_FLAT_BID
        bid_lane0 = p10dc_orbitcta_flat_bid(
            nn ? D_BKF_HIGH_NN_OFF : D_BKF_HIGH_NRNL_OFF,
            base_off, nblocks, q);
#endif
    }
    const uint32_t q = __shfl_sync(active, q_lane0, 0);
    const bool nn = __shfl_sync(active, nn_lane0, 0) != 0u;
    const P10DCHighClosureCompactPreCtx* const z =
        p10dc_forward_compact_prectx_warpcoop_ptr(q, nn);
#if P10DC_RANKFORMULA_PRECTX_FLAT_BID
    const uint32_t desc_meta = p10dc_compact_prectx_warpcoop_load_meta(z);
    const uint32_t bid = desc_meta >> 24;
#else
    const uint32_t desc_meta = 0u;
    const uint32_t bid = __shfl_sync(active, bid_lane0, 0);
#endif

    uint32_t valid_lane0 = 0u;
    if (lane == 0u && bid < nblocks) {
        const BucketOrbitOp op = nn ? D_BKF_HIGH_NN[q] : D_BKF_HIGH_NRNL[q];
        const uint32_t sl = bkf_orbit_src(op);
        const uint32_t jl = bkf_orbit_partner(op);
        const uint32_t dl = bkf_orbit_drop(op);
        const uint32_t ss = bkf_loc_owner(sl);
        const uint32_t js = bkf_loc_owner(jl);
        const uint32_t ds = bkf_loc_owner(dl);
        c.xb = bkf_high_main(ss, bid);
        if (c.xb.valid && c.xb.rows && c.xb.cols) {
            uint32_t jbid = bid;
            if (p == LOW_LUT_K + 1) {
                const uint32_t center = nn ? uint32_t(R) : uint32_t(N);
                const int he = int(c.xb.hs) + (center == uint32_t(R) ? 1 : 0);
                jbid = uint32_t(3 * he + int(center));
            }
            c.jb = bkf_high_main(js, jbid);
            c.db = bkf_high_block(ds, uint32_t(c.xb.hs));
            p10dc_direct_resolve_high_io(
                c, ss, js, ds,
                bkf_loc_rank(sl), bkf_loc_rank(jl), bkf_loc_rank(dl));
            c.kind = uint8_t(nn ? CPU_ORBIT_NN : CPU_ORBIT_NR);
            c.valid = 1;
            valid_lane0 = 1u;
        }
    }
    const bool valid = __shfl_sync(active, valid_lane0, 0) != 0u;
    if (valid) {
#if P10DC_RANKFORMULA_PRECTX_FLAT_BID
        p10dc_apply_compact_prectx_warpcoop_ptr_meta(c, z, desc_meta);
#else
        p10dc_apply_compact_prectx_warpcoop_ptr(c, z);
#endif
    }
}

__device__ __forceinline__ void
p10dc_orbitcta_flat_dynamic_pipe2_prepare_reverse_producer_prectx_warpcoop(
    P10DC_ORBITCTA_CTX& c,
    uint32_t k_lane0,
    uint32_t base_off,
    const uint32_t* stream_base,
    uint32_t nblocks,
    int p,
    bool edge
) {
    const uint32_t lane = uint32_t(threadIdx.x) & 31u;
    const unsigned active = __activemask();
    const uint32_t k = __shfl_sync(active, k_lane0, 0);
    if (k == 0xffffffffu) return;

    uint32_t q_lane0 = 0u;
    uint32_t kind_lane0 = 0u;
    BucketOrbitOp op_lane0{};
#if !P10DC_RANKFORMULA_PRECTX_FLAT_BID
    uint32_t bid_lane0 = 0xffffffffu;
#endif
    if (lane == 0u) {
        c.valid = 0;
        uint32_t q = 0u, kind = 0u;
        const uint32_t* off = nullptr;
        BucketOrbitOp op{};
        if (k < c.n0) {
            kind = CPU_ORBIT_NN; off = D_RS54_HIGH_NN_OFF;
            q = stream_base[0] + k;
            op = D_RS54_HIGH_NN[q];
        } else if (k < c.n0 + c.n1) {
            kind = CPU_ORBIT_NR; off = D_RS54_HIGH_NR_OFF;
            q = stream_base[1] + k - c.n0;
            op = D_RS54_HIGH_NR[q];
        } else {
            kind = CPU_ORBIT_NL; off = D_RS54_HIGH_NL_OFF;
            q = stream_base[2] + k - c.n0 - c.n1;
            op = D_RS54_HIGH_NL[q];
        }
        q_lane0 = q;
        kind_lane0 = kind;
        op_lane0 = op;
#if !P10DC_RANKFORMULA_PRECTX_FLAT_BID
        bid_lane0 = p10dc_orbitcta_flat_bid(off, base_off, nblocks, q);
#endif
    }
    const uint32_t q = __shfl_sync(active, q_lane0, 0);
    const uint32_t kind = __shfl_sync(active, kind_lane0, 0);
    const P10DCHighClosureCompactPreCtx* const z =
        p10dc_reverse_compact_prectx_warpcoop_ptr(q, kind);
#if P10DC_RANKFORMULA_PRECTX_FLAT_BID
    const uint32_t desc_meta = p10dc_compact_prectx_warpcoop_load_meta(z);
    const uint32_t bid = desc_meta >> 24;
#else
    const uint32_t desc_meta = 0u;
    const uint32_t bid = __shfl_sync(active, bid_lane0, 0);
#endif

    uint32_t valid_lane0 = 0u;
    if (lane == 0u && bid < nblocks) {
        const BucketOrbitOp op = op_lane0;
        const uint32_t sl = bkf_orbit_src(op);
        const uint32_t jl = bkf_orbit_partner(op);
        const uint32_t dl = bkf_orbit_drop(op);
        const uint32_t ss = bkf_loc_owner(sl);
        const uint32_t js = bkf_loc_owner(jl);
        const uint32_t ds = bkf_loc_owner(dl);
        c.xb = bkf_high_main(ss, bid);
        if (c.xb.valid && c.xb.rows && c.xb.cols) {
            c.jb = bkf_high_main(js, bkcp10_reverse_high_jblock(bid, c.xb, p, kind));
            c.db = bkf_high_block(ds, uint32_t(c.xb.hs));
            p10dc_direct_resolve_high_io(
                c, ss, js, ds,
                bkf_loc_rank(sl), bkf_loc_rank(jl), bkf_loc_rank(dl));
            c.kind = uint8_t(kind);
            c.valid = 1;
            valid_lane0 = 1u;
        }
        (void)edge;
    }
    const bool valid = __shfl_sync(active, valid_lane0, 0) != 0u;
    if (valid) {
#if P10DC_RANKFORMULA_PRECTX_FLAT_BID
        p10dc_apply_compact_prectx_warpcoop_ptr_meta(c, z, desc_meta);
#else
        p10dc_apply_compact_prectx_warpcoop_ptr(c, z);
#endif
    }
}
