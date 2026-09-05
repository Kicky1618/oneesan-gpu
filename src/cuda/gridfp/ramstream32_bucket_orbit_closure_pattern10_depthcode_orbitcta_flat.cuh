#pragma once

#include "ramstream32_bucket_orbit_closure_pattern10_depthcode_warpctx.cuh"

#ifndef P10DC_ORBITCTA_CTX
#define P10DC_ORBITCTA_CTX P10DCHighResolvedCtx
#define P10DC_ORBITCTA_CTX_LOCAL 1
#endif
#ifndef P10DC_ORBITCTA_PREPARE_FORWARD
#define P10DC_ORBITCTA_PREPARE_FORWARD(c,payload,loc,p,ss,js,ds,sr,jr,dr) do { \
    (c).plan = p10dc_forward_high((payload),(loc),(c).db,(p)); \
    p10dc_resolve_high_rows((c),(ss),(js),(ds),(sr),(jr),(dr)); \
} while(0)
#define P10DC_ORBITCTA_PREPARE_FORWARD_LOCAL 1
#endif
#ifndef P10DC_ORBITCTA_PREPARE_REVERSE
#define P10DC_ORBITCTA_PREPARE_REVERSE(c,payload,loc,plan_db,p,edge,ss,js,ds,sr,jr,dr) do { \
    (c).plan = p10dc_reverse_high((payload),(loc),(plan_db),(p),(edge)); \
    p10dc_resolve_high_rows((c),(ss),(js),(ds),(sr),(jr),(dr)); \
} while(0)
#define P10DC_ORBITCTA_PREPARE_REVERSE_LOCAL 1
#endif
#ifndef P10DC_ORBITCTA_PLAN_SUM
#define P10DC_ORBITCTA_PLAN_SUM(c,db,lr) p10dc_resolved_high_plan_sum((c),(db),(lr))
#define P10DC_ORBITCTA_PLAN_SUM_LOCAL 1
#endif
#ifndef P10DC_ORBITCTA_EARLY_JP
#define P10DC_ORBITCTA_EARLY_JP 0
#endif
#ifndef P10DC_ORBITCTA_COL_ILP
#define P10DC_ORBITCTA_COL_ILP 1
#endif
static_assert(P10DC_ORBITCTA_COL_ILP == 1 || P10DC_ORBITCTA_COL_ILP == 2 ||
              P10DC_ORBITCTA_COL_ILP == 4,
              "P10DC_ORBITCTA_COL_ILP must be 1, 2, or 4");

// q belongs to the unique non-empty bucket interval [off[b],off[b+1]).
// Empty buckets create repeated offsets, so choose the last boundary <= q.
__device__ __forceinline__ uint32_t p10dc_orbitcta_flat_bid(
    const uint32_t* off, uint32_t base, uint32_t nblocks, uint32_t q
) {
    uint32_t lo = 0, hi = nblocks;
#pragma unroll 6
    for (int it = 0; it < 6; ++it) {
        if (lo + 1u >= hi) break;
        const uint32_t mid = (lo + hi) >> 1;
        if (off[base + mid] <= q) lo = mid;
        else hi = mid;
    }
    return lo;
}

__device__ __forceinline__ void p10dc_orbitcta_flat_forward_columns(
    P10DC_ORBITCTA_CTX& c
) {
    constexpr int ILP = P10DC_ORBITCTA_COL_ILP;
    const uint32_t cols = c.xb.cols;
    Count* const ip_base = c.ip_base;
    Count* const jp_base = c.jp_base;
    Count* const dp_base = c.dp_base;
    const uint32_t kind = c.kind;
    const uint32_t lane_step = uint32_t(blockDim.x);
    const uint32_t group_step = lane_step * uint32_t(ILP);
    for (uint32_t base = uint32_t(threadIdx.x); base < cols; base += group_step) {
        uint32_t lr[ILP]{};
        uint8_t valid[ILP]{};
        Count x[ILP]{}, old[ILP]{}, y[ILP]{}, extra[ILP]{};
#pragma unroll
        for (int j = 0; j < ILP; ++j) {
            lr[j] = base + uint32_t(j) * lane_step;
            valid[j] = uint8_t(lr[j] < cols);
            if (valid[j]) {
                x[j] = ip_base[lr[j]];
                old[j] = dp_base[lr[j]];
#if P10DC_ORBITCTA_EARLY_JP
                y[j] = jp_base[lr[j]];
#endif
            }
        }
#ifdef P10DC_ORBITCTA_PLAN_SUM_PAIR
#pragma unroll
        for (int j = 0; j < ILP; j += 2) {
            if constexpr (ILP > 1) {
                if (valid[j] && valid[j + 1]) {
                    P10DC_ORBITCTA_PLAN_SUM_PAIR(c, c.db, lr[j], lr[j + 1], extra[j], extra[j + 1]);
                } else {
                    if (valid[j]) extra[j] = P10DC_ORBITCTA_PLAN_SUM(c, c.db, lr[j]);
                    if (valid[j + 1]) extra[j + 1] = P10DC_ORBITCTA_PLAN_SUM(c, c.db, lr[j + 1]);
                }
            } else if (valid[j]) {
                extra[j] = P10DC_ORBITCTA_PLAN_SUM(c, c.db, lr[j]);
            }
        }
#else
#pragma unroll
        for (int j = 0; j < ILP; ++j)
            if (valid[j]) extra[j] = P10DC_ORBITCTA_PLAN_SUM(c, c.db, lr[j]);
#endif
#pragma unroll
        for (int j = 0; j < ILP; ++j) {
            if (!valid[j]) continue;
            if (kind == CPU_ORBIT_NN) {
#if P10DC_ORBITCTA_EARLY_JP
                jp_base[lr[j]] = gpu_direct_add(y[j], x[j]);
#else
                jp_base[lr[j]] = gpu_direct_add(jp_base[lr[j]], x[j]);
#endif
                ip_base[lr[j]] = gpu_direct_add(x[j], old[j]);
                dp_base[lr[j]] = extra[j];
            } else {
#if !P10DC_ORBITCTA_EARLY_JP
                y[j] = jp_base[lr[j]];
#endif
                ip_base[lr[j]] = gpu_direct_add(gpu_direct_add(x[j], y[j]), old[j]);
                dp_base[lr[j]] = gpu_direct_add(x[j], extra[j]);
            }
        }
    }
}

__device__ __forceinline__ void p10dc_orbitcta_flat_reverse_columns(
    P10DC_ORBITCTA_CTX& c, bool edge
) {
    constexpr int ILP = P10DC_ORBITCTA_COL_ILP;
    const uint32_t cols = c.xb.cols;
    Count* const ip_base = c.ip_base;
    Count* const jp_base = c.jp_base;
    Count* const dp_base = c.dp_base;
    const uint32_t kind = c.kind;
    const uint32_t lane_step = uint32_t(blockDim.x);
    const uint32_t group_step = lane_step * uint32_t(ILP);
    const BucketPhysicalBlock& sum_db = edge ? c.xb : c.db;
    for (uint32_t base = uint32_t(threadIdx.x); base < cols; base += group_step) {
        uint32_t lr[ILP]{};
        uint8_t valid[ILP]{};
        Count x[ILP]{}, old[ILP]{}, y[ILP]{}, extra[ILP]{};
#pragma unroll
        for (int j = 0; j < ILP; ++j) {
            lr[j] = base + uint32_t(j) * lane_step;
            valid[j] = uint8_t(lr[j] < cols);
            if (valid[j]) {
                x[j] = ip_base[lr[j]];
                old[j] = dp_base[lr[j]];
#if P10DC_ORBITCTA_EARLY_JP
                y[j] = jp_base[lr[j]];
#endif
            }
        }
#ifdef P10DC_ORBITCTA_PLAN_SUM_PAIR
#pragma unroll
        for (int j = 0; j < ILP; j += 2) {
            if constexpr (ILP > 1) {
                if (valid[j] && valid[j + 1]) {
                    P10DC_ORBITCTA_PLAN_SUM_PAIR(c, sum_db, lr[j], lr[j + 1], extra[j], extra[j + 1]);
                } else {
                    if (valid[j]) extra[j] = P10DC_ORBITCTA_PLAN_SUM(c, sum_db, lr[j]);
                    if (valid[j + 1]) extra[j + 1] = P10DC_ORBITCTA_PLAN_SUM(c, sum_db, lr[j + 1]);
                }
            } else if (valid[j]) {
                extra[j] = P10DC_ORBITCTA_PLAN_SUM(c, sum_db, lr[j]);
            }
        }
#else
#pragma unroll
        for (int j = 0; j < ILP; ++j)
            if (valid[j]) extra[j] = P10DC_ORBITCTA_PLAN_SUM(c, sum_db, lr[j]);
#endif
#pragma unroll
        for (int j = 0; j < ILP; ++j) {
            if (!valid[j]) continue;
            if (kind == CPU_ORBIT_NN) {
#if P10DC_ORBITCTA_EARLY_JP
                jp_base[lr[j]] = gpu_direct_add(y[j], x[j]);
#else
                jp_base[lr[j]] = gpu_direct_add(jp_base[lr[j]], x[j]);
#endif
                ip_base[lr[j]] = gpu_direct_add(gpu_direct_add(x[j], old[j]), edge ? extra[j] : 0);
                dp_base[lr[j]] = edge ? 0 : extra[j];
            } else {
#if !P10DC_ORBITCTA_EARLY_JP
                y[j] = jp_base[lr[j]];
#endif
                ip_base[lr[j]] = gpu_direct_add(gpu_direct_add(x[j], y[j]), old[j]);
                if (edge) {
                    jp_base[lr[j]] = gpu_direct_add(x[j], y[j]);
                    dp_base[lr[j]] = 0;
                } else {
                    dp_base[lr[j]] = gpu_direct_add(x[j], extra[j]);
                }
            }
        }
    }
}

// One global persistent CTA pool spans all HIGH buckets for this position.
// This removes per-bucket Y tails and the grid.z*grid.y empty-CTA product.
__global__ void bucket_high_orbit_closure_pattern10_depthcode_orbitcta_flat_kernel(int p) {
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
            const uint32_t bid = p10dc_orbitcta_flat_bid(
                nn ? D_BKF_HIGH_NN_OFF : D_BKF_HIGH_NRNL_OFF,
                base_off, nblocks, q);
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
                const uint32_t payload = p10dc_payload(op, false, true, sid, p, uint32_t(c.xb.hs));
                P10DC_ORBITCTA_PREPARE_FORWARD(
                    c, payload, dl, p, ss, js, ds,
                    bkf_loc_rank(sl), bkf_loc_rank(jl), bkf_loc_rank(dl));
                c.kind = uint8_t(nn ? CPU_ORBIT_NN : CPU_ORBIT_NR);
                c.valid = 1;
            }
        }
        __syncthreads();
        if (c.valid) p10dc_orbitcta_flat_forward_columns(c);
        __syncthreads();
    }
}

__global__ void bucket_reverse_high_pattern10_depthcode_orbitcta_flat_kernel(int p) {
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
            const uint32_t *off = nullptr;
            BucketOrbitOp op;
            if (k < c.n0) {
                kind = CPU_ORBIT_NN; sid = 0u; off = D_RS54_HIGH_NN_OFF;
                q = D_RS54_HIGH_NN_OFF[base_off] + k;
                op = D_RS54_HIGH_NN[q];
            } else if (k < c.n0 + c.n1) {
                kind = CPU_ORBIT_NR; sid = 1u; off = D_RS54_HIGH_NR_OFF;
                q = D_RS54_HIGH_NR_OFF[base_off] + k - c.n0;
                op = D_RS54_HIGH_NR[q];
            } else {
                kind = CPU_ORBIT_NL; sid = 2u; off = D_RS54_HIGH_NL_OFF;
                q = D_RS54_HIGH_NL_OFF[base_off] + k - c.n0 - c.n1;
                op = D_RS54_HIGH_NL[q];
            }
            const uint32_t bid = p10dc_orbitcta_flat_bid(off, base_off, nblocks, q);
            const uint32_t sl = bkf_orbit_src(op), jl = bkf_orbit_partner(op), dl = bkf_orbit_drop(op);
            const uint32_t ss = bkf_loc_owner(sl), js = bkf_loc_owner(jl), ds = bkf_loc_owner(dl);
            c.xb = bkf_high_main(ss, bid);
            if (c.xb.valid && c.xb.rows && c.xb.cols) {
                c.jb = bkf_high_main(js, bkcp10_reverse_high_jblock(bid, c.xb, p, kind));
                c.db = bkf_high_block(ds, uint32_t(c.xb.hs));
                const uint32_t payload = p10dc_payload(op, true, true, sid, p, uint32_t(c.xb.hs));
                P10DC_ORBITCTA_PREPARE_REVERSE(
                    c, payload, edge ? sl : dl, edge ? c.xb : c.db, p, edge,
                    ss, js, ds, bkf_loc_rank(sl), bkf_loc_rank(jl), bkf_loc_rank(dl));
                c.kind = uint8_t(kind);
                c.valid = 1;
            }
        }
        __syncthreads();
        if (c.valid) p10dc_orbitcta_flat_reverse_columns(c, edge);
        __syncthreads();
    }
}

#ifdef P10DC_ORBITCTA_PLAN_SUM_LOCAL
#undef P10DC_ORBITCTA_PLAN_SUM_LOCAL
#undef P10DC_ORBITCTA_PLAN_SUM
#endif
#ifdef P10DC_ORBITCTA_PREPARE_REVERSE_LOCAL
#undef P10DC_ORBITCTA_PREPARE_REVERSE_LOCAL
#undef P10DC_ORBITCTA_PREPARE_REVERSE
#endif
#ifdef P10DC_ORBITCTA_PREPARE_FORWARD_LOCAL
#undef P10DC_ORBITCTA_PREPARE_FORWARD_LOCAL
#undef P10DC_ORBITCTA_PREPARE_FORWARD
#endif
#ifdef P10DC_ORBITCTA_CTX_LOCAL
#undef P10DC_ORBITCTA_CTX_LOCAL
#undef P10DC_ORBITCTA_CTX
#endif
