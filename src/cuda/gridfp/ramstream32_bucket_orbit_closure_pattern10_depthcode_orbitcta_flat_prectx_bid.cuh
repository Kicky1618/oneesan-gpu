#pragma once

#ifndef P10DC_ORBITCTA_CTX
#error "flat prectx-bid scheduler requires P10DC_ORBITCTA_CTX"
#endif
#ifndef P10DC_ORBITCTA_PREPARE_FORWARD
#error "flat prectx-bid scheduler requires forward prepare hook"
#endif
#ifndef P10DC_ORBITCTA_PREPARE_REVERSE
#error "flat prectx-bid scheduler requires reverse prepare hook"
#endif
#if !P10DC_RANKFORMULA_PRECTX_COMPACT || !P10DC_RANKFORMULA_PRECTX_FORWARD || !P10DC_RANKFORMULA_PRECTX_REVERSE
#error "flat prectx-bid scheduler requires compact forward+reverse prectx"
#endif

// Exact flat scheduler with no q->bucket offset search. The existing compact
// prectx padding byte carries the source main bid, so each orbit keeps the
// original one-orbit cyclic load distribution while deleting the 6-step
// binary search from thread-0 setup.
__global__ void bucket_high_orbit_closure_pattern10_depthcode_orbitcta_flat_prectx_bid_kernel(int p) {
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

    for (uint32_t k = uint32_t(blockIdx.x); k < c.total; k += uint32_t(gridDim.x)) {
        if (threadIdx.x == 0) {
            c.valid = 0;
            const bool nn = k < c.n0;
            const uint32_t q = nn
                ? D_BKF_HIGH_NN_OFF[base_off] + k
                : c.n1 + k - c.n0;
            const uint32_t bid = p10dc_forward_compact_prectx_flat_bid(q, nn);
            if (bid < nblocks) {
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
                    const uint32_t payload = p10dc_payload(
                        op, false, true, sid, p, uint32_t(c.xb.hs));
                    P10DC_ORBITCTA_PREPARE_FORWARD(
                        c, payload, dl, p, ss, js, ds,
                        bkf_loc_rank(sl), bkf_loc_rank(jl), bkf_loc_rank(dl));
                    c.kind = uint8_t(nn ? CPU_ORBIT_NN : CPU_ORBIT_NR);
                    c.valid = 1;
                }
            }
        }
        __syncthreads();
        if (c.valid) p10dc_orbitcta_flat_forward_columns(c);
        __syncthreads();
    }
}

__global__ void bucket_reverse_high_pattern10_depthcode_orbitcta_flat_prectx_bid_kernel(int p) {
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

    for (uint32_t k = uint32_t(blockIdx.x); k < c.total; k += uint32_t(gridDim.x)) {
        if (threadIdx.x == 0) {
            c.valid = 0;
            uint32_t q = 0, kind = 0, sid = 0;
            BucketOrbitOp op;
            if (k < c.n0) {
                kind = CPU_ORBIT_NN; sid = 0u;
                q = D_RS54_HIGH_NN_OFF[base_off] + k;
                op = D_RS54_HIGH_NN[q];
            } else if (k < c.n0 + c.n1) {
                kind = CPU_ORBIT_NR; sid = 1u;
                q = D_RS54_HIGH_NR_OFF[base_off] + k - c.n0;
                op = D_RS54_HIGH_NR[q];
            } else {
                kind = CPU_ORBIT_NL; sid = 2u;
                q = D_RS54_HIGH_NL_OFF[base_off] + k - c.n0 - c.n1;
                op = D_RS54_HIGH_NL[q];
            }
            const uint32_t bid = p10dc_reverse_compact_prectx_flat_bid(q, kind);
            if (bid < nblocks) {
                const uint32_t sl = bkf_orbit_src(op), jl = bkf_orbit_partner(op), dl = bkf_orbit_drop(op);
                const uint32_t ss = bkf_loc_owner(sl), js = bkf_loc_owner(jl), ds = bkf_loc_owner(dl);
                c.xb = bkf_high_main(ss, bid);
                if (c.xb.valid && c.xb.rows && c.xb.cols) {
                    c.jb = bkf_high_main(js, bkcp10_reverse_high_jblock(bid, c.xb, p, kind));
                    c.db = bkf_high_block(ds, uint32_t(c.xb.hs));
                    const uint32_t payload = p10dc_payload(
                        op, true, true, sid, p, uint32_t(c.xb.hs));
                    P10DC_ORBITCTA_PREPARE_REVERSE(
                        c, payload, edge ? sl : dl, edge ? c.xb : c.db, p, edge,
                        ss, js, ds, bkf_loc_rank(sl), bkf_loc_rank(jl), bkf_loc_rank(dl));
                    c.kind = uint8_t(kind);
                    c.valid = 1;
                }
            }
        }
        __syncthreads();
        if (c.valid) p10dc_orbitcta_flat_reverse_columns(c, edge);
        __syncthreads();
    }
}
