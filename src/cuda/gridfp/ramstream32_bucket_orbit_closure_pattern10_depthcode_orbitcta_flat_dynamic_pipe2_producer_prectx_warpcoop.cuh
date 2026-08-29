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

// Called by warp 0 only. k_lane0 is meaningful only in lane 0 and is broadcast
// with a shuffle. Lane 0 resolves orbit ownership and the three direct I/O rows;
// lanes 0..8 then expand the already-built compact closure descriptor in
// parallel. No warp barrier is required afterwards: dynamic pipe2 already ends
// each current-orbit consumer phase with a CTA barrier before publishing next.
// If compact flat-bid is installed, the cached pad byte replaces the six-step
// flat bucket binary search without changing the distributed descriptor decode.
__device__ __forceinline__ void
p10dc_orbitcta_flat_dynamic_pipe2_prepare_forward_producer_prectx_warpcoop(
    P10DC_ORBITCTA_CTX& c,
    uint32_t k_lane0,
    uint32_t base_off,
    uint32_t nblocks,
    int p
) {
    const uint32_t lane = uint32_t(threadIdx.x) & 31u;
    const unsigned active = __activemask();
    const uint32_t k = __shfl_sync(active, k_lane0, 0);
    if (k == 0xffffffffu) return;

    uint32_t q_lane0 = 0u;
    uint32_t meta_lane0 = 0u;
    if (lane == 0u) {
        c.valid = 0;
        const bool nn = k < c.n0;
        const uint32_t q = nn
            ? D_BKF_HIGH_NN_OFF[base_off] + k
            : c.n1 + k - c.n0;
#if P10DC_RANKFORMULA_PRECTX_FLAT_BID
        const uint32_t bid = p10dc_forward_compact_prectx_flat_bid(q, nn);
#else
        const uint32_t bid = p10dc_orbitcta_flat_bid(
            nn ? D_BKF_HIGH_NN_OFF : D_BKF_HIGH_NRNL_OFF,
            base_off, nblocks, q);
#endif
        if (bid < nblocks) {
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
                q_lane0 = q;
                meta_lane0 = 1u | (uint32_t(c.kind) << 8);
            }
        }
    }

    const uint32_t q = __shfl_sync(active, q_lane0, 0);
    const uint32_t meta = __shfl_sync(active, meta_lane0, 0);
    if (meta & 1u)
        p10dc_apply_forward_compact_prectx_warpcoop(
            c, q, ((meta >> 8) & 0xffu) == uint32_t(CPU_ORBIT_NN));
}

__device__ __forceinline__ void
p10dc_orbitcta_flat_dynamic_pipe2_prepare_reverse_producer_prectx_warpcoop(
    P10DC_ORBITCTA_CTX& c,
    uint32_t k_lane0,
    uint32_t base_off,
    uint32_t nblocks,
    int p,
    bool edge
) {
    const uint32_t lane = uint32_t(threadIdx.x) & 31u;
    const unsigned active = __activemask();
    const uint32_t k = __shfl_sync(active, k_lane0, 0);
    if (k == 0xffffffffu) return;

    uint32_t q_lane0 = 0u;
    uint32_t meta_lane0 = 0u;
    if (lane == 0u) {
        c.valid = 0;
        uint32_t q = 0u, kind = 0u;
#if !P10DC_RANKFORMULA_PRECTX_FLAT_BID
        const uint32_t* off = nullptr;
#endif
        BucketOrbitOp op{};
        if (k < c.n0) {
            kind = CPU_ORBIT_NN;
#if !P10DC_RANKFORMULA_PRECTX_FLAT_BID
            off = D_RS54_HIGH_NN_OFF;
#endif
            q = D_RS54_HIGH_NN_OFF[base_off] + k;
            op = D_RS54_HIGH_NN[q];
        } else if (k < c.n0 + c.n1) {
            kind = CPU_ORBIT_NR;
#if !P10DC_RANKFORMULA_PRECTX_FLAT_BID
            off = D_RS54_HIGH_NR_OFF;
#endif
            q = D_RS54_HIGH_NR_OFF[base_off] + k - c.n0;
            op = D_RS54_HIGH_NR[q];
        } else {
            kind = CPU_ORBIT_NL;
#if !P10DC_RANKFORMULA_PRECTX_FLAT_BID
            off = D_RS54_HIGH_NL_OFF;
#endif
            q = D_RS54_HIGH_NL_OFF[base_off] + k - c.n0 - c.n1;
            op = D_RS54_HIGH_NL[q];
        }
#if P10DC_RANKFORMULA_PRECTX_FLAT_BID
        const uint32_t bid = p10dc_reverse_compact_prectx_flat_bid(q, kind);
#else
        const uint32_t bid = p10dc_orbitcta_flat_bid(off, base_off, nblocks, q);
#endif
        if (bid < nblocks) {
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
                q_lane0 = q;
                meta_lane0 = 1u | (kind << 8);
            }
        }
        (void)edge;
    }

    const uint32_t q = __shfl_sync(active, q_lane0, 0);
    const uint32_t meta = __shfl_sync(active, meta_lane0, 0);
    if (meta & 1u)
        p10dc_apply_reverse_compact_prectx_warpcoop(c, q, (meta >> 8) & 0xffu);
}
