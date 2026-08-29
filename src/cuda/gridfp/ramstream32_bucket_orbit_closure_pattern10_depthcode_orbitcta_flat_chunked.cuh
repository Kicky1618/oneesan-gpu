#pragma once

// Included while the rankformula orbit-CTA hook macros are still active and
// after orbitcta_flat.cuh has provided the common column executors + flat_bid.
#ifndef P10DC_ORBITCTA_CTX
#error "flat chunked scheduler requires P10DC_ORBITCTA_CTX"
#endif
#ifndef P10DC_ORBITCTA_PREPARE_FORWARD
#error "flat chunked scheduler requires forward prepare hook"
#endif
#ifndef P10DC_ORBITCTA_PREPARE_REVERSE
#error "flat chunked scheduler requires reverse prepare hook"
#endif
#ifndef P10DC_ORBITCTA_FLAT_CHUNK
#define P10DC_ORBITCTA_FLAT_CHUNK 1
#endif
#ifndef P10DC_RANKFORMULA_PRECTX_WARPCOOP
#define P10DC_RANKFORMULA_PRECTX_WARPCOOP 0
#endif
static_assert(P10DC_ORBITCTA_FLAT_CHUNK >= 2 && P10DC_ORBITCTA_FLAT_CHUNK <= 32,
              "chunked flat scheduler expects chunk size 2..32");
static_assert(P10DC_RANKFORMULA_PRECTX_WARPCOOP == 0 ||
              P10DC_RANKFORMULA_PRECTX_WARPCOOP == 1,
              "P10DC_RANKFORMULA_PRECTX_WARPCOOP must be 0 or 1");
#if P10DC_RANKFORMULA_PRECTX_WARPCOOP
#include "ramstream32_bucket_precomputed_high_ctx_compact_warpcoop.cuh"
static_assert(P10DC_RANKFORMULA_PRECTX_COMPACT == 1 &&
              P10DC_RANKFORMULA_PRECTX_FORWARD == 1 &&
              P10DC_RANKFORMULA_PRECTX_REVERSE == 1,
              "warp-cooperative chunked prectx requires compact forward+reverse prectx");
#endif

__device__ __forceinline__ uint32_t p10dc_orbitcta_flat_advance_bid(
    const uint32_t* off, uint32_t base, uint32_t nblocks,
    uint32_t q, uint32_t bid
) {
    // q only advances by one inside a local chunk. Repeated boundaries are
    // empty buckets, so walk across all of them until q belongs to
    // [off[bid], off[bid+1]).
    while (bid + 1u < nblocks && off[base + bid + 1u] <= q) ++bid;
    return bid;
}

// Chunk-cyclic persistent scheduler: CTA b owns CHUNK consecutive orbit ids,
// then jumps by gridDim.x*CHUNK. The first orbit of a chunk pays the 6-step
// bucket binary search; following orbits advance through bucket boundaries.
// This retains cyclic load spreading while amortizing bucket lookup traffic.
__global__ void bucket_high_orbit_closure_pattern10_depthcode_orbitcta_flat_chunked_kernel(int p) {
    const uint32_t nblocks = D_BKF_MAIN_NBLOCKS;
    if (!nblocks) return;
    const uint32_t pi = uint32_t((TARGET_W - 1) - p);
    const uint32_t base_off = pi * D_BKF_HIGH_PITCH;
    extern __shared__ unsigned long long storage[];
    P10DC_ORBITCTA_CTX& c = *reinterpret_cast<P10DC_ORBITCTA_CTX*>(storage);

    if (threadIdx.x == 0) {
        const uint32_t nn0 = D_BKF_HIGH_NN_OFF[base_off];
        const uint32_t nn1 = D_BKF_HIGH_NN_OFF[base_off + nblocks];
        const uint32_t nr0 = D_BKF_HIGH_NRNL_OFF[base_off];
        const uint32_t nr1 = D_BKF_HIGH_NRNL_OFF[base_off + nblocks];
        c.n0 = nn1 - nn0;
        c.n1 = nr0;
        c.total = c.n0 + (nr1 - nr0);
    }
    __syncthreads();

    constexpr uint32_t CHUNK = uint32_t(P10DC_ORBITCTA_FLAT_CHUNK);
    const uint32_t chunk_stride = uint32_t(gridDim.x) * CHUNK;
    for (uint32_t chunk0 = uint32_t(blockIdx.x) * CHUNK;
         chunk0 < c.total; chunk0 += chunk_stride) {
        uint32_t cached_stream = 0xffffffffu;
        uint32_t cached_bid = 0u;
#pragma unroll 1
        for (uint32_t j = 0; j < CHUNK; ++j) {
            const uint32_t k = chunk0 + j;
            if (k >= c.total) break;
            if (threadIdx.x == 0) {
                c.valid = 0;
                const bool nn = k < c.n0;
                const uint32_t stream = nn ? 0u : 1u;
                const uint32_t q = nn
                    ? D_BKF_HIGH_NN_OFF[base_off] + k
                    : c.n1 + k - c.n0;
                const uint32_t* off = nn ? D_BKF_HIGH_NN_OFF : D_BKF_HIGH_NRNL_OFF;
                if (stream != cached_stream) {
                    cached_bid = p10dc_orbitcta_flat_bid(off, base_off, nblocks, q);
                    cached_stream = stream;
                } else {
                    cached_bid = p10dc_orbitcta_flat_advance_bid(
                        off, base_off, nblocks, q, cached_bid);
                }
                const uint32_t bid = cached_bid;
                const BucketOrbitOp op = nn ? D_BKF_HIGH_NN[q] : D_BKF_HIGH_NRNL[q];
                const uint32_t sid = nn ? 0u : 3u;
                const uint32_t sl = bkf_orbit_src(op), jl = bkf_orbit_partner(op), dl = bkf_orbit_drop(op);
                const uint32_t ss = bkf_loc_owner(sl), js = bkf_loc_owner(jl), ds = bkf_loc_owner(dl);
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
                    c.kind = uint8_t(nn ? CPU_ORBIT_NN : CPU_ORBIT_NR);
#if P10DC_RANKFORMULA_PRECTX_WARPCOOP
                    // Direct I/O rows are independent of closure prectx. Publish
                    // q temporarily through cross_depth; warp 0 replaces it with
                    // the real CROSS depth after resolving compact row refs.
                    p10dc_direct_resolve_high_io(
                        c, ss, js, ds,
                        bkf_loc_rank(sl), bkf_loc_rank(jl), bkf_loc_rank(dl));
                    c.cross_depth = q;
#else
                    const uint32_t payload = p10dc_payload(
                        op, false, true, sid, p, uint32_t(c.xb.hs));
                    P10DC_ORBITCTA_PREPARE_FORWARD(
                        c, payload, dl, p, ss, js, ds,
                        bkf_loc_rank(sl), bkf_loc_rank(jl), bkf_loc_rank(dl));
#endif
                    c.valid = 1;
                }
            }
#if P10DC_RANKFORMULA_PRECTX_WARPCOOP
            if (threadIdx.x < 32u) {
                // First sync publishes lane-0 setup. Capture q/kind in every
                // warp-0 lane before lane BKCZ_MAX_LOCAL overwrites cross_depth.
                __syncwarp();
                const uint32_t coop_q = c.cross_depth;
                const uint32_t coop_kind = uint32_t(c.kind);
                const uint32_t coop_valid = uint32_t(c.valid);
                __syncwarp();
                if (coop_valid)
                    p10dc_apply_forward_compact_prectx_warpcoop(
                        c, coop_q, coop_kind == uint32_t(CPU_ORBIT_NN));
            }
#endif
            __syncthreads();
            if (c.valid) p10dc_orbitcta_flat_forward_columns(c);
            __syncthreads();
        }
    }
}

__global__ void bucket_reverse_high_pattern10_depthcode_orbitcta_flat_chunked_kernel(int p) {
    const uint32_t nblocks = D_BKF_MAIN_NBLOCKS;
    if (!nblocks) return;
    const uint32_t pi = uint32_t(p - (LOW_LUT_K + 1));
    const uint32_t base_off = pi * D_RS54_PITCH;
    const bool edge = p == TARGET_W - 1;
    extern __shared__ unsigned long long storage[];
    P10DC_ORBITCTA_CTX& c = *reinterpret_cast<P10DC_ORBITCTA_CTX*>(storage);

    if (threadIdx.x == 0) {
        const uint32_t nn0 = D_RS54_HIGH_NN_OFF[base_off];
        const uint32_t nn1 = D_RS54_HIGH_NN_OFF[base_off + nblocks];
        const uint32_t nr0 = D_RS54_HIGH_NR_OFF[base_off];
        const uint32_t nr1 = D_RS54_HIGH_NR_OFF[base_off + nblocks];
        const uint32_t nl0 = D_RS54_HIGH_NL_OFF[base_off];
        const uint32_t nl1 = D_RS54_HIGH_NL_OFF[base_off + nblocks];
        c.n0 = nn1 - nn0;
        c.n1 = nr1 - nr0;
        c.total = c.n0 + c.n1 + (nl1 - nl0);
    }
    __syncthreads();

    constexpr uint32_t CHUNK = uint32_t(P10DC_ORBITCTA_FLAT_CHUNK);
    const uint32_t chunk_stride = uint32_t(gridDim.x) * CHUNK;
    for (uint32_t chunk0 = uint32_t(blockIdx.x) * CHUNK;
         chunk0 < c.total; chunk0 += chunk_stride) {
        uint32_t cached_stream = 0xffffffffu;
        uint32_t cached_bid = 0u;
#pragma unroll 1
        for (uint32_t j = 0; j < CHUNK; ++j) {
            const uint32_t k = chunk0 + j;
            if (k >= c.total) break;
            if (threadIdx.x == 0) {
                c.valid = 0;
                uint32_t q = 0, kind = 0, sid = 0, stream = 0;
                const uint32_t* off = nullptr;
                BucketOrbitOp op;
                if (k < c.n0) {
                    stream = 0u; kind = CPU_ORBIT_NN; sid = 0u; off = D_RS54_HIGH_NN_OFF;
                    q = D_RS54_HIGH_NN_OFF[base_off] + k;
                    op = D_RS54_HIGH_NN[q];
                } else if (k < c.n0 + c.n1) {
                    stream = 1u; kind = CPU_ORBIT_NR; sid = 1u; off = D_RS54_HIGH_NR_OFF;
                    q = D_RS54_HIGH_NR_OFF[base_off] + k - c.n0;
                    op = D_RS54_HIGH_NR[q];
                } else {
                    stream = 2u; kind = CPU_ORBIT_NL; sid = 2u; off = D_RS54_HIGH_NL_OFF;
                    q = D_RS54_HIGH_NL_OFF[base_off] + k - c.n0 - c.n1;
                    op = D_RS54_HIGH_NL[q];
                }
                if (stream != cached_stream) {
                    cached_bid = p10dc_orbitcta_flat_bid(off, base_off, nblocks, q);
                    cached_stream = stream;
                } else {
                    cached_bid = p10dc_orbitcta_flat_advance_bid(
                        off, base_off, nblocks, q, cached_bid);
                }
                const uint32_t bid = cached_bid;
                const uint32_t sl = bkf_orbit_src(op), jl = bkf_orbit_partner(op), dl = bkf_orbit_drop(op);
                const uint32_t ss = bkf_loc_owner(sl), js = bkf_loc_owner(jl), ds = bkf_loc_owner(dl);
                c.xb = bkf_high_main(ss, bid);
                if (c.xb.valid && c.xb.rows && c.xb.cols) {
                    c.jb = bkf_high_main(js, bkcp10_reverse_high_jblock(bid, c.xb, p, kind));
                    c.db = bkf_high_block(ds, uint32_t(c.xb.hs));
                    c.kind = uint8_t(kind);
#if P10DC_RANKFORMULA_PRECTX_WARPCOOP
                    p10dc_direct_resolve_high_io(
                        c, ss, js, ds,
                        bkf_loc_rank(sl), bkf_loc_rank(jl), bkf_loc_rank(dl));
                    c.cross_depth = q;
#else
                    const uint32_t payload = p10dc_payload(
                        op, true, true, sid, p, uint32_t(c.xb.hs));
                    P10DC_ORBITCTA_PREPARE_REVERSE(
                        c, payload, edge ? sl : dl, edge ? c.xb : c.db, p, edge,
                        ss, js, ds, bkf_loc_rank(sl), bkf_loc_rank(jl), bkf_loc_rank(dl));
#endif
                    c.valid = 1;
                }
            }
#if P10DC_RANKFORMULA_PRECTX_WARPCOOP
            if (threadIdx.x < 32u) {
                __syncwarp();
                const uint32_t coop_q = c.cross_depth;
                const uint32_t coop_kind = uint32_t(c.kind);
                const uint32_t coop_valid = uint32_t(c.valid);
                __syncwarp();
                if (coop_valid)
                    p10dc_apply_reverse_compact_prectx_warpcoop(
                        c, coop_q, coop_kind);
            }
#endif
            __syncthreads();
            if (c.valid) p10dc_orbitcta_flat_reverse_columns(c, edge);
            __syncthreads();
        }
    }
}
