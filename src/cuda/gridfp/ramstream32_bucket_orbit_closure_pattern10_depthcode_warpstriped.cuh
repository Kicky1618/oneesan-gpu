#pragma once

#include "ramstream32_bucket_orbit_closure_pattern10_depthcode_warpctx.cuh"

// EXPERIMENTAL: assign a distinct HIGH orbit to each warp instead of having
// every warp resolve the same orbit. This removes duplicated payload decode,
// plan construction and row setup, but it is only correct if simultaneously
// scheduled orbit footprints are independent. BucketPhysicalBlock ranges may
// overlap, so schedule coverage alone is not a proof: write/write and
// write/closure-read row-address overlap must be checked before exact use.
// Each warp covers one orbit with 32-column stripes.
static inline bool p10dc_warpstriped_threads_ok(int threads) {
    return threads > 0 && threads <= 1024 && (threads & 31) == 0;
}

#ifndef P10DC_WARPSTRIPED_EARLY_JP_LOAD
#define P10DC_WARPSTRIPED_EARLY_JP_LOAD 0
#endif
#ifndef P10DC_WARPSTRIPED_COL_ILP
#define P10DC_WARPSTRIPED_COL_ILP 1
#endif
static_assert(P10DC_WARPSTRIPED_EARLY_JP_LOAD == 0 || P10DC_WARPSTRIPED_EARLY_JP_LOAD == 1,
              "P10DC_WARPSTRIPED_EARLY_JP_LOAD must be 0 or 1");
static_assert(P10DC_WARPSTRIPED_COL_ILP == 1 || P10DC_WARPSTRIPED_COL_ILP == 2 ||
              P10DC_WARPSTRIPED_COL_ILP == 4,
              "P10DC_WARPSTRIPED_COL_ILP must be 1, 2, or 4");

#ifndef P10DC_WARPSTRIPED_CTX
#define P10DC_WARPSTRIPED_CTX P10DCHighResolvedCtx
#define P10DC_WARPSTRIPED_CTX_LOCAL 1
#endif
#ifndef P10DC_WARPSTRIPED_PREPARE_FORWARD
#define P10DC_WARPSTRIPED_PREPARE_FORWARD(c,payload,loc,p,ss,js,ds,sr,jr,dr) do { \
    (c).plan = p10dc_forward_high((payload),(loc),(c).db,(p)); \
    p10dc_resolve_high_rows((c),(ss),(js),(ds),(sr),(jr),(dr)); \
} while(0)
#define P10DC_WARPSTRIPED_PREPARE_FORWARD_LOCAL 1
#endif
#ifndef P10DC_WARPSTRIPED_PREPARE_REVERSE
#define P10DC_WARPSTRIPED_PREPARE_REVERSE(c,payload,loc,plan_db,p,edge,ss,js,ds,sr,jr,dr) do { \
    (c).plan = p10dc_reverse_high((payload),(loc),(plan_db),(p),(edge)); \
    p10dc_resolve_high_rows((c),(ss),(js),(ds),(sr),(jr),(dr)); \
} while(0)
#define P10DC_WARPSTRIPED_PREPARE_REVERSE_LOCAL 1
#endif

__global__ void bucket_high_orbit_closure_pattern10_depthcode_warpstriped_kernel(int p) {
    uint32_t bid = blockIdx.z;
    if (bid >= D_BKF_MAIN_NBLOCKS) return;
    uint32_t pi = uint32_t((TARGET_W - 1) - p);
    uint32_t oi = uint32_t(size_t(pi) * D_BKF_HIGH_PITCH + bid);
    extern __shared__ unsigned long long warp_ctx_storage[];
    auto* warp_ctx = reinterpret_cast<P10DC_WARPSTRIPED_CTX*>(warp_ctx_storage);

    const uint32_t lane = uint32_t(threadIdx.x) & 31u;
    const uint32_t warp = uint32_t(threadIdx.x) >> 5;
    const uint32_t nwarps = uint32_t(blockDim.x) >> 5;
    const unsigned active = __activemask();
    P10DC_WARPSTRIPED_CTX& c = warp_ctx[warp];

    if (lane == 0) {
        uint32_t na = D_BKF_HIGH_NN_OFF[oi], nb = D_BKF_HIGH_NN_OFF[oi + 1];
        uint32_t ra = D_BKF_HIGH_NRNL_OFF[oi], rb = D_BKF_HIGH_NRNL_OFF[oi + 1];
        c.n0 = nb - na;
        c.n1 = ra;
        c.total = c.n0 + (rb - ra);
    }
    __syncwarp(active);
    const uint32_t total = c.total;

    for (uint32_t k = uint32_t(blockIdx.y) * nwarps + warp;
         k < total;
         k += uint32_t(gridDim.y) * nwarps) {
        if (lane == 0) {
            c.valid = 0;
            bool nn = k < c.n0;
            uint32_t qi = nn ? D_BKF_HIGH_NN_OFF[oi] + k : c.n1 + k - c.n0;
            uint32_t sid = nn ? 0u : 3u;
            BucketOrbitOp op = nn ? D_BKF_HIGH_NN[qi] : D_BKF_HIGH_NRNL[qi];
            uint32_t sl = bkf_orbit_src(op), jl = bkf_orbit_partner(op), dl = bkf_orbit_drop(op);
            uint32_t ss = bkf_loc_owner(sl), js = bkf_loc_owner(jl), ds = bkf_loc_owner(dl);
            c.xb = bkf_high_main(ss, bid);
            if (c.xb.valid && c.xb.rows && c.xb.cols) {
                uint32_t jbid = bid;
                if (p == LOW_LUT_K + 1) {
                    uint32_t center = nn ? uint32_t(R) : uint32_t(N);
                    int he = int(c.xb.hs) + (center == uint32_t(R) ? 1 : 0);
                    jbid = uint32_t(3 * he + int(center));
                }
                c.jb = bkf_high_main(js, jbid);
                c.db = bkf_high_block(ds, uint32_t(c.xb.hs));
                uint32_t payload = p10dc_payload(op, false, true, sid, p, uint32_t(c.xb.hs));
                P10DC_WARPSTRIPED_PREPARE_FORWARD(
                    c, payload, dl, p, ss, js, ds,
                    bkf_loc_rank(sl), bkf_loc_rank(jl), bkf_loc_rank(dl));
                c.kind = uint8_t(nn ? CPU_ORBIT_NN : CPU_ORBIT_NR);
                c.valid = 1;
            }
        }
        __syncwarp(active);

        if (c.valid) {
            const BucketPhysicalBlock xb = c.xb;
            const BucketPhysicalBlock db = c.db;
            Count* const ip_base = c.ip_base;
            Count* const jp_base = c.jp_base;
            Count* const dp_base = c.dp_base;
            const uint32_t kind = c.kind;
            const uint32_t step = uint32_t(gridDim.x) * 32u;
            for (uint32_t base = uint32_t(blockIdx.x) * 32u + lane;
                 base < xb.cols;
                 base += step * uint32_t(P10DC_WARPSTRIPED_COL_ILP)) {
                uint32_t lr[P10DC_WARPSTRIPED_COL_ILP];
                Count x[P10DC_WARPSTRIPED_COL_ILP]{};
                Count old[P10DC_WARPSTRIPED_COL_ILP]{};
                Count y[P10DC_WARPSTRIPED_COL_ILP]{};
                Count extra[P10DC_WARPSTRIPED_COL_ILP]{};
                bool live[P10DC_WARPSTRIPED_COL_ILP]{};
#pragma unroll
                for (int t = 0; t < P10DC_WARPSTRIPED_COL_ILP; ++t) {
                    lr[t] = base + uint32_t(t) * step;
                    live[t] = lr[t] < xb.cols;
                    if (live[t]) {
                        x[t] = ip_base[lr[t]];
                        old[t] = dp_base[lr[t]];
#if P10DC_WARPSTRIPED_EARLY_JP_LOAD
                        y[t] = jp_base[lr[t]];
#endif
                    }
                }
#ifdef P10DC_WARPSTRIPED_PLAN_SUM_QUAD
                if (live[3]) {
                    P10DC_WARPSTRIPED_PLAN_SUM_QUAD(
                        c, db, lr[0], lr[1], lr[2], lr[3],
                        extra[0], extra[1], extra[2], extra[3]);
                } else
#endif
                {
#ifdef P10DC_WARPSTRIPED_PLAN_SUM_PAIR
#pragma unroll
                    for (int t = 0; t < P10DC_WARPSTRIPED_COL_ILP; t += 2) {
                        if (!live[t]) continue;
                        if (live[t + 1]) {
                            P10DC_WARPSTRIPED_PLAN_SUM_PAIR(
                                c, db, lr[t], lr[t + 1], extra[t], extra[t + 1]);
                        } else {
                            extra[t] = p10dc_resolved_high_plan_sum(c, db, lr[t]);
                        }
                    }
#else
#pragma unroll
                    for (int t = 0; t < P10DC_WARPSTRIPED_COL_ILP; ++t) {
                        if (live[t]) extra[t] = p10dc_resolved_high_plan_sum(c, db, lr[t]);
                    }
#endif
                }
#pragma unroll
                for (int t = 0; t < P10DC_WARPSTRIPED_COL_ILP; ++t) {
                    if (!live[t]) continue;
                    Count* ip = ip_base + lr[t];
                    Count* jp = jp_base + lr[t];
                    Count* dp = dp_base + lr[t];
                    if (kind == CPU_ORBIT_NN) {
#if P10DC_WARPSTRIPED_EARLY_JP_LOAD
                        *jp = gpu_direct_add(y[t], x[t]);
#else
                        *jp = gpu_direct_add(*jp, x[t]);
#endif
                        *ip = gpu_direct_add(x[t], old[t]);
                        *dp = extra[t];
                    } else {
#if !P10DC_WARPSTRIPED_EARLY_JP_LOAD
                        y[t] = *jp;
#endif
                        *ip = gpu_direct_add(gpu_direct_add(x[t], y[t]), old[t]);
                        *dp = gpu_direct_add(x[t], extra[t]);
                    }
                }
            }
        }
        __syncwarp(active);
    }
}

__global__ void bucket_reverse_high_pattern10_depthcode_warpstriped_kernel(int p) {
    uint32_t bid = blockIdx.z;
    if (bid >= D_BKF_MAIN_NBLOCKS) return;
    uint32_t pi = uint32_t(p - (LOW_LUT_K + 1));
    uint32_t oi = uint32_t(size_t(pi) * D_RS54_PITCH + bid);
    const bool edge = p == TARGET_W - 1;
    extern __shared__ unsigned long long warp_ctx_storage[];
    auto* warp_ctx = reinterpret_cast<P10DC_WARPSTRIPED_CTX*>(warp_ctx_storage);

    const uint32_t lane = uint32_t(threadIdx.x) & 31u;
    const uint32_t warp = uint32_t(threadIdx.x) >> 5;
    const uint32_t nwarps = uint32_t(blockDim.x) >> 5;
    const unsigned active = __activemask();
    P10DC_WARPSTRIPED_CTX& c = warp_ctx[warp];

    if (lane == 0) {
        uint32_t na = D_RS54_HIGH_NN_OFF[oi], nb = D_RS54_HIGH_NN_OFF[oi + 1];
        uint32_t ra = D_RS54_HIGH_NR_OFF[oi], rb = D_RS54_HIGH_NR_OFF[oi + 1];
        uint32_t la = D_RS54_HIGH_NL_OFF[oi], lb = D_RS54_HIGH_NL_OFF[oi + 1];
        c.n0 = nb - na;
        c.n1 = rb - ra;
        c.total = c.n0 + c.n1 + (lb - la);
    }
    __syncwarp(active);
    const uint32_t total = c.total;

    for (uint32_t k = uint32_t(blockIdx.y) * nwarps + warp;
         k < total;
         k += uint32_t(gridDim.y) * nwarps) {
        if (lane == 0) {
            c.valid = 0;
            uint32_t qi = 0, kind = 0, sid = 0;
            BucketOrbitOp op;
            if (k < c.n0) {
                kind = CPU_ORBIT_NN; sid = 0;
                qi = D_RS54_HIGH_NN_OFF[oi] + k;
                op = D_RS54_HIGH_NN[qi];
            } else if (k < c.n0 + c.n1) {
                kind = CPU_ORBIT_NR; sid = 1;
                qi = D_RS54_HIGH_NR_OFF[oi] + k - c.n0;
                op = D_RS54_HIGH_NR[qi];
            } else {
                kind = CPU_ORBIT_NL; sid = 2;
                qi = D_RS54_HIGH_NL_OFF[oi] + k - c.n0 - c.n1;
                op = D_RS54_HIGH_NL[qi];
            }
            uint32_t sl = bkf_orbit_src(op), jl = bkf_orbit_partner(op), dl = bkf_orbit_drop(op);
            uint32_t ss = bkf_loc_owner(sl), js = bkf_loc_owner(jl), ds = bkf_loc_owner(dl);
            c.xb = bkf_high_main(ss, bid);
            if (c.xb.valid && c.xb.rows && c.xb.cols) {
                c.jb = bkf_high_main(js, bkcp10_reverse_high_jblock(bid, c.xb, p, kind));
                c.db = bkf_high_block(ds, uint32_t(c.xb.hs));
                uint32_t payload = p10dc_payload(op, true, true, sid, p, uint32_t(c.xb.hs));
                P10DC_WARPSTRIPED_PREPARE_REVERSE(
                    c, payload, edge ? sl : dl, edge ? c.xb : c.db, p, edge,
                    ss, js, ds, bkf_loc_rank(sl), bkf_loc_rank(jl), bkf_loc_rank(dl));
                c.kind = uint8_t(kind);
                c.valid = 1;
            }
        }
        __syncwarp(active);

        if (c.valid) {
            const BucketPhysicalBlock xb = c.xb;
            const BucketPhysicalBlock db = c.db;
            Count* const ip_base = c.ip_base;
            Count* const jp_base = c.jp_base;
            Count* const dp_base = c.dp_base;
            const uint32_t kind = c.kind;
            const uint32_t step = uint32_t(gridDim.x) * 32u;
            for (uint32_t base = uint32_t(blockIdx.x) * 32u + lane;
                 base < xb.cols;
                 base += step * uint32_t(P10DC_WARPSTRIPED_COL_ILP)) {
                uint32_t lr[P10DC_WARPSTRIPED_COL_ILP];
                Count x[P10DC_WARPSTRIPED_COL_ILP]{};
                Count old[P10DC_WARPSTRIPED_COL_ILP]{};
                Count y[P10DC_WARPSTRIPED_COL_ILP]{};
                Count extra[P10DC_WARPSTRIPED_COL_ILP]{};
                bool live[P10DC_WARPSTRIPED_COL_ILP]{};
#pragma unroll
                for (int t = 0; t < P10DC_WARPSTRIPED_COL_ILP; ++t) {
                    lr[t] = base + uint32_t(t) * step;
                    live[t] = lr[t] < xb.cols;
                    if (live[t]) {
                        x[t] = ip_base[lr[t]];
                        old[t] = dp_base[lr[t]];
#if P10DC_WARPSTRIPED_EARLY_JP_LOAD
                        y[t] = jp_base[lr[t]];
#endif
                    }
                }
#ifdef P10DC_WARPSTRIPED_PLAN_SUM_QUAD
                if (live[3]) {
                    P10DC_WARPSTRIPED_PLAN_SUM_QUAD(
                        c, edge ? xb : db, lr[0], lr[1], lr[2], lr[3],
                        extra[0], extra[1], extra[2], extra[3]);
                } else
#endif
                {
#ifdef P10DC_WARPSTRIPED_PLAN_SUM_PAIR
#pragma unroll
                    for (int t = 0; t < P10DC_WARPSTRIPED_COL_ILP; t += 2) {
                        if (!live[t]) continue;
                        if (live[t + 1]) {
                            P10DC_WARPSTRIPED_PLAN_SUM_PAIR(
                                c, edge ? xb : db, lr[t], lr[t + 1], extra[t], extra[t + 1]);
                        } else {
                            extra[t] = p10dc_resolved_high_plan_sum(c, edge ? xb : db, lr[t]);
                        }
                    }
#else
#pragma unroll
                    for (int t = 0; t < P10DC_WARPSTRIPED_COL_ILP; ++t) {
                        if (live[t]) extra[t] = p10dc_resolved_high_plan_sum(c, edge ? xb : db, lr[t]);
                    }
#endif
                }
#pragma unroll
                for (int t = 0; t < P10DC_WARPSTRIPED_COL_ILP; ++t) {
                    if (!live[t]) continue;
                    Count* ip = ip_base + lr[t];
                    Count* jp = jp_base + lr[t];
                    Count* dp = dp_base + lr[t];
                    if (kind == CPU_ORBIT_NN) {
#if P10DC_WARPSTRIPED_EARLY_JP_LOAD
                        *jp = gpu_direct_add(y[t], x[t]);
#else
                        *jp = gpu_direct_add(*jp, x[t]);
#endif
                        *ip = gpu_direct_add(gpu_direct_add(x[t], old[t]), edge ? extra[t] : 0);
                        *dp = edge ? 0 : extra[t];
                    } else {
#if !P10DC_WARPSTRIPED_EARLY_JP_LOAD
                        y[t] = *jp;
#endif
                        *ip = gpu_direct_add(gpu_direct_add(x[t], y[t]), old[t]);
                        if (edge) {
                            *jp = gpu_direct_add(x[t], y[t]);
                            *dp = 0;
                        } else {
                            *dp = gpu_direct_add(x[t], extra[t]);
                        }
                    }
                }
            }
        }
        __syncwarp(active);
    }
}

#ifdef P10DC_WARPSTRIPED_PREPARE_REVERSE_LOCAL
#undef P10DC_WARPSTRIPED_PREPARE_REVERSE_LOCAL
#undef P10DC_WARPSTRIPED_PREPARE_REVERSE
#endif
#ifdef P10DC_WARPSTRIPED_PREPARE_FORWARD_LOCAL
#undef P10DC_WARPSTRIPED_PREPARE_FORWARD_LOCAL
#undef P10DC_WARPSTRIPED_PREPARE_FORWARD
#endif
#ifdef P10DC_WARPSTRIPED_CTX_LOCAL
#undef P10DC_WARPSTRIPED_CTX_LOCAL
#undef P10DC_WARPSTRIPED_CTX
#endif
