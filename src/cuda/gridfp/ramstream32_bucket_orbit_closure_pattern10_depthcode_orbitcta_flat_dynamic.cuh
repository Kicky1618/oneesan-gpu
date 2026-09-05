#pragma once

#ifndef P10DC_ORBITCTA_CTX
#error "flat dynamic scheduler requires P10DC_ORBITCTA_CTX"
#endif
#ifndef P10DC_ORBITCTA_PREPARE_FORWARD
#error "flat dynamic scheduler requires forward prepare hook"
#endif
#ifndef P10DC_ORBITCTA_PREPARE_REVERSE
#error "flat dynamic scheduler requires reverse prepare hook"
#endif
#ifndef P10DC_RANKFORMULA_PRECTX_FLAT_BID
#define P10DC_RANKFORMULA_PRECTX_FLAT_BID 0
#endif
#ifndef P10DC_RANKFORMULA_PRECTX_FLAT_BID_FUSED
#define P10DC_RANKFORMULA_PRECTX_FLAT_BID_FUSED 0
#endif
#ifndef P10DC_ORBITCTA_FLAT_DYNAMIC_BATCH
#define P10DC_ORBITCTA_FLAT_DYNAMIC_BATCH 1
#endif
#ifndef P10DC_ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP
#define P10DC_ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP 0
#endif
#ifndef P10DC_ORBITCTA_FLAT_DYNAMIC_ADAPTIVE_WAVES
#define P10DC_ORBITCTA_FLAT_DYNAMIC_ADAPTIVE_WAVES 0
#endif
static_assert(P10DC_ORBITCTA_FLAT_DYNAMIC_BATCH == 1 ||
              P10DC_ORBITCTA_FLAT_DYNAMIC_BATCH == 2 ||
              P10DC_ORBITCTA_FLAT_DYNAMIC_BATCH == 4 ||
              P10DC_ORBITCTA_FLAT_DYNAMIC_BATCH == 8 ||
              P10DC_ORBITCTA_FLAT_DYNAMIC_BATCH == 16,
              "dynamic flat queue batch must be 1,2,4,8,16");
static_assert(P10DC_ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP == 0 ||
              P10DC_ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP == 1,
              "dynamic lease/prepare fusion must be 0 or 1");
static_assert(P10DC_ORBITCTA_FLAT_DYNAMIC_ADAPTIVE_WAVES == 0 ||
              P10DC_ORBITCTA_FLAT_DYNAMIC_ADAPTIVE_WAVES == 1 ||
              P10DC_ORBITCTA_FLAT_DYNAMIC_ADAPTIVE_WAVES == 2 ||
              P10DC_ORBITCTA_FLAT_DYNAMIC_ADAPTIVE_WAVES == 4,
              "dynamic adaptive waves must be 0,1,2,4");

// One 32-bit queue head per device. Graph capture inserts a reset kernel before
// every HIGH position. Each resident CTA atomically leases BATCH consecutive
// orbit ids, then processes that lease locally. BATCH=1 is the pure dynamic
// queue; larger batches reduce atomic contention at the cost of a bounded
// BATCH-orbit scheduling tail.
__device__ uint32_t D_P10DC_ORBITCTA_FLAT_NEXT = 0;

__global__ void p10dc_orbitcta_flat_dynamic_reset_kernel() {
    if (blockIdx.x == 0 && threadIdx.x == 0) D_P10DC_ORBITCTA_FLAT_NEXT = 0;
}

__device__ __forceinline__ uint32_t p10dc_orbitcta_flat_dynamic_effective_batch(uint32_t total) {
    constexpr uint32_t MAX_BATCH = uint32_t(P10DC_ORBITCTA_FLAT_DYNAMIC_BATCH);
#if P10DC_ORBITCTA_FLAT_DYNAMIC_ADAPTIVE_WAVES == 0
    (void)total;
    return MAX_BATCH;
#else
    // Keep at least ADAPTIVE_WAVES lease waves available across the launched
    // persistent CTA pool. This shrinks a large compile-time batch only for
    // small HIGH positions where fixed batching would strand resident CTAs.
    constexpr uint32_t WAVES = uint32_t(P10DC_ORBITCTA_FLAT_DYNAMIC_ADAPTIVE_WAVES);
    const uint64_t denom = uint64_t(gridDim.x) * uint64_t(WAVES);
    if (!denom) return MAX_BATCH;
    const uint32_t cap = uint32_t((uint64_t(total) + denom - 1u) / denom);
    uint32_t b = 1;
    if (MAX_BATCH >= 2 && cap >= 2) b = 2;
    if (MAX_BATCH >= 4 && cap >= 4) b = 4;
    if (MAX_BATCH >= 8 && cap >= 8) b = 8;
    if (MAX_BATCH >= 16 && cap >= 16) b = 16;
    return b;
#endif
}

__device__ __forceinline__ void p10dc_orbitcta_flat_dynamic_prepare_forward(
    P10DC_ORBITCTA_CTX& c,
    uint32_t k,
    uint32_t base_off,
    uint32_t nblocks,
    int p
) {
    const bool nn = k < c.n0;
    const uint32_t q = nn
        ? D_BKF_HIGH_NN_OFF[base_off] + k
        : c.n1 + k - c.n0;
#if P10DC_RANKFORMULA_PRECTX_FLAT_BID
#if P10DC_RANKFORMULA_PRECTX_FLAT_BID_FUSED
    const P10DCHighClosureCompactPreCtx z =
        p10dc_load_forward_compact_prectx_flat(q, nn);
    const uint32_t bid = uint32_t(z.pad);
#else
    const uint32_t bid = p10dc_forward_compact_prectx_flat_bid(q, nn);
#endif
#else
    const uint32_t bid = p10dc_orbitcta_flat_bid(
        nn ? D_BKF_HIGH_NN_OFF : D_BKF_HIGH_NRNL_OFF,
        base_off, nblocks, q);
#endif
    if (bid >= nblocks) return;
    const BucketOrbitOp op = nn ? D_BKF_HIGH_NN[q] : D_BKF_HIGH_NRNL[q];
    const uint32_t sl = bkf_orbit_src(op), jl = bkf_orbit_partner(op), dl = bkf_orbit_drop(op);
    const uint32_t ss = bkf_loc_owner(sl), js = bkf_loc_owner(jl), ds = bkf_loc_owner(dl);
    c.xb = bkf_high_main(ss, bid);
    if (!(c.xb.valid && c.xb.rows && c.xb.cols)) return;
    uint32_t jbid = bid;
    if (p == LOW_LUT_K + 1) {
        const uint32_t center = nn ? uint32_t(R) : uint32_t(N);
        const int he = int(c.xb.hs) + (center == uint32_t(R) ? 1 : 0);
        jbid = uint32_t(3 * he + int(center));
    }
    c.jb = bkf_high_main(js, jbid);
    c.db = bkf_high_block(ds, uint32_t(c.xb.hs));
#if P10DC_RANKFORMULA_PRECTX_FLAT_BID
    p10dc_direct_resolve_high_io(
        c, ss, js, ds,
        bkf_loc_rank(sl), bkf_loc_rank(jl), bkf_loc_rank(dl));
#if P10DC_RANKFORMULA_PRECTX_FLAT_BID_FUSED
    p10dc_apply_loaded_compact_prectx(c, z);
#else
    p10dc_apply_forward_prectx(c, q, nn);
#endif
#else
    const uint32_t sid = nn ? 0u : 3u;
    const uint32_t payload = p10dc_payload(op, false, true, sid, p, uint32_t(c.xb.hs));
    P10DC_ORBITCTA_PREPARE_FORWARD(
        c, payload, dl, p, ss, js, ds,
        bkf_loc_rank(sl), bkf_loc_rank(jl), bkf_loc_rank(dl));
#endif
    c.kind = uint8_t(nn ? CPU_ORBIT_NN : CPU_ORBIT_NR);
    c.valid = 1;
}

__device__ __forceinline__ void p10dc_orbitcta_flat_dynamic_prepare_reverse(
    P10DC_ORBITCTA_CTX& c,
    uint32_t k,
    uint32_t base_off,
    uint32_t nblocks,
    int p,
    bool edge
) {
    uint32_t q = 0, kind = 0;
#if !P10DC_RANKFORMULA_PRECTX_FLAT_BID
    uint32_t sid = 0;
    const uint32_t* off = nullptr;
#endif
    BucketOrbitOp op;
    if (k < c.n0) {
        kind = CPU_ORBIT_NN;
#if !P10DC_RANKFORMULA_PRECTX_FLAT_BID
        sid = 0u; off = D_RS54_HIGH_NN_OFF;
#endif
        q = D_RS54_HIGH_NN_OFF[base_off] + k;
        op = D_RS54_HIGH_NN[q];
    } else if (k < c.n0 + c.n1) {
        kind = CPU_ORBIT_NR;
#if !P10DC_RANKFORMULA_PRECTX_FLAT_BID
        sid = 1u; off = D_RS54_HIGH_NR_OFF;
#endif
        q = D_RS54_HIGH_NR_OFF[base_off] + k - c.n0;
        op = D_RS54_HIGH_NR[q];
    } else {
        kind = CPU_ORBIT_NL;
#if !P10DC_RANKFORMULA_PRECTX_FLAT_BID
        sid = 2u; off = D_RS54_HIGH_NL_OFF;
#endif
        q = D_RS54_HIGH_NL_OFF[base_off] + k - c.n0 - c.n1;
        op = D_RS54_HIGH_NL[q];
    }
#if P10DC_RANKFORMULA_PRECTX_FLAT_BID
#if P10DC_RANKFORMULA_PRECTX_FLAT_BID_FUSED
    const P10DCHighClosureCompactPreCtx z =
        p10dc_load_reverse_compact_prectx_flat(q, kind);
    const uint32_t bid = uint32_t(z.pad);
#else
    const uint32_t bid = p10dc_reverse_compact_prectx_flat_bid(q, kind);
#endif
#else
    const uint32_t bid = p10dc_orbitcta_flat_bid(off, base_off, nblocks, q);
#endif
    if (bid >= nblocks) return;
    const uint32_t sl = bkf_orbit_src(op), jl = bkf_orbit_partner(op), dl = bkf_orbit_drop(op);
    const uint32_t ss = bkf_loc_owner(sl), js = bkf_loc_owner(jl), ds = bkf_loc_owner(dl);
    c.xb = bkf_high_main(ss, bid);
    if (!(c.xb.valid && c.xb.rows && c.xb.cols)) return;
    c.jb = bkf_high_main(js, bkcp10_reverse_high_jblock(bid, c.xb, p, kind));
    c.db = bkf_high_block(ds, uint32_t(c.xb.hs));
#if P10DC_RANKFORMULA_PRECTX_FLAT_BID
    p10dc_direct_resolve_high_io(
        c, ss, js, ds,
        bkf_loc_rank(sl), bkf_loc_rank(jl), bkf_loc_rank(dl));
#if P10DC_RANKFORMULA_PRECTX_FLAT_BID_FUSED
    p10dc_apply_loaded_compact_prectx(c, z);
#else
    p10dc_apply_reverse_prectx(c, q, kind);
#endif
#else
    const uint32_t payload = p10dc_payload(op, true, true, sid, p, uint32_t(c.xb.hs));
    P10DC_ORBITCTA_PREPARE_REVERSE(
        c, payload, edge ? sl : dl, edge ? c.xb : c.db, p, edge,
        ss, js, ds, bkf_loc_rank(sl), bkf_loc_rank(jl), bkf_loc_rank(dl));
#endif
    c.kind = uint8_t(kind);
    c.valid = 1;
}

__global__ void bucket_high_orbit_closure_pattern10_depthcode_orbitcta_flat_dynamic_kernel(int p) {
    const uint32_t nblocks = D_BKF_MAIN_NBLOCKS;
    if (!nblocks) return;
    const uint32_t pi = uint32_t((TARGET_W - 1) - p);
    const uint32_t base_off = pi * D_BKF_HIGH_PITCH;
    extern __shared__ unsigned long long storage[];
    __shared__ uint32_t lease_base;
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
    constexpr uint32_t BATCH = uint32_t(P10DC_ORBITCTA_FLAT_DYNAMIC_BATCH);
#if P10DC_ORBITCTA_FLAT_DYNAMIC_ADAPTIVE_WAVES == 0
    constexpr uint32_t LEASE_BATCH = BATCH;
#else
    const uint32_t LEASE_BATCH = p10dc_orbitcta_flat_dynamic_effective_batch(c.total);
#endif
    for (;;) {
#if P10DC_ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP
        if (threadIdx.x == 0) {
            lease_base = atomicAdd(&D_P10DC_ORBITCTA_FLAT_NEXT, LEASE_BATCH);
            c.valid = 0;
            if (lease_base < c.total)
                p10dc_orbitcta_flat_dynamic_prepare_forward(c, lease_base, base_off, nblocks, p);
        }
        // One CTA barrier publishes both the lease id and the first prepared
        // orbit. The baseline path needs one barrier for the lease and another
        // for j=0 preparation.
        __syncthreads();
        if (lease_base >= c.total) break;
        if (c.valid == 1) p10dc_orbitcta_flat_forward_columns(c);
        __syncthreads();
#pragma unroll 1
        for (uint32_t j = 1; j < LEASE_BATCH; ++j) {
            if (threadIdx.x == 0) {
                c.valid = 0;
                const uint32_t k = lease_base + j;
                if (k < c.total)
                    p10dc_orbitcta_flat_dynamic_prepare_forward(c, k, base_off, nblocks, p);
            }
            __syncthreads();
            if (c.valid == 1) p10dc_orbitcta_flat_forward_columns(c);
            __syncthreads();
        }
#else
        if (threadIdx.x == 0)
            lease_base = atomicAdd(&D_P10DC_ORBITCTA_FLAT_NEXT, LEASE_BATCH);
        __syncthreads();
        if (lease_base >= c.total) break;
#pragma unroll 1
        for (uint32_t j = 0; j < LEASE_BATCH; ++j) {
            if (threadIdx.x == 0) {
                c.valid = 0;
                const uint32_t k = lease_base + j;
                if (k < c.total)
                    p10dc_orbitcta_flat_dynamic_prepare_forward(c, k, base_off, nblocks, p);
            }
            __syncthreads();
            if (c.valid == 1) p10dc_orbitcta_flat_forward_columns(c);
            __syncthreads();
        }
#endif
    }
}

__global__ void bucket_reverse_high_pattern10_depthcode_orbitcta_flat_dynamic_kernel(int p) {
    const uint32_t nblocks = D_BKF_MAIN_NBLOCKS;
    if (!nblocks) return;
    const uint32_t pi = uint32_t(p - (LOW_LUT_K + 1));
    const uint32_t base_off = pi * D_RS54_PITCH;
    const bool edge = p == TARGET_W - 1;
    extern __shared__ unsigned long long storage[];
    __shared__ uint32_t lease_base;
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
    constexpr uint32_t BATCH = uint32_t(P10DC_ORBITCTA_FLAT_DYNAMIC_BATCH);
#if P10DC_ORBITCTA_FLAT_DYNAMIC_ADAPTIVE_WAVES == 0
    constexpr uint32_t LEASE_BATCH = BATCH;
#else
    const uint32_t LEASE_BATCH = p10dc_orbitcta_flat_dynamic_effective_batch(c.total);
#endif
    for (;;) {
#if P10DC_ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP
        if (threadIdx.x == 0) {
            lease_base = atomicAdd(&D_P10DC_ORBITCTA_FLAT_NEXT, LEASE_BATCH);
            c.valid = 0;
            if (lease_base < c.total)
                p10dc_orbitcta_flat_dynamic_prepare_reverse(
                    c, lease_base, base_off, nblocks, p, edge);
        }
        __syncthreads();
        if (lease_base >= c.total) break;
        if (c.valid == 1) p10dc_orbitcta_flat_reverse_columns(c, edge);
        __syncthreads();
#pragma unroll 1
        for (uint32_t j = 1; j < LEASE_BATCH; ++j) {
            if (threadIdx.x == 0) {
                c.valid = 0;
                const uint32_t k = lease_base + j;
                if (k < c.total)
                    p10dc_orbitcta_flat_dynamic_prepare_reverse(
                        c, k, base_off, nblocks, p, edge);
            }
            __syncthreads();
            if (c.valid == 1) p10dc_orbitcta_flat_reverse_columns(c, edge);
            __syncthreads();
        }
#else
        if (threadIdx.x == 0)
            lease_base = atomicAdd(&D_P10DC_ORBITCTA_FLAT_NEXT, LEASE_BATCH);
        __syncthreads();
        if (lease_base >= c.total) break;
#pragma unroll 1
        for (uint32_t j = 0; j < LEASE_BATCH; ++j) {
            if (threadIdx.x == 0) {
                c.valid = 0;
                const uint32_t k = lease_base + j;
                if (k < c.total)
                    p10dc_orbitcta_flat_dynamic_prepare_reverse(c, k, base_off, nblocks, p, edge);
            }
            __syncthreads();
            if (c.valid == 1) p10dc_orbitcta_flat_reverse_columns(c, edge);
            __syncthreads();
        }
#endif
    }
}
