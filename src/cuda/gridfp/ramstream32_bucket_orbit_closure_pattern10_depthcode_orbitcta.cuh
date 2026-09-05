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

// Bandwidth-oriented HIGH scheduler. One CTA owns one orbit/context. Column
// ILP batches independent LOW-rank columns per thread. An optional pair-plan
// hook lets descriptor/source gathers for adjacent columns overlap directly.
__global__ void bucket_high_orbit_closure_pattern10_depthcode_orbitcta_kernel(int p) {
    if (blockIdx.x) return;
    const uint32_t bid = blockIdx.z;
    if (bid >= D_BKF_MAIN_NBLOCKS) return;
    const uint32_t pi = uint32_t((TARGET_W - 1) - p);
    const uint32_t oi = uint32_t(size_t(pi) * D_BKF_HIGH_PITCH + bid);
    extern __shared__ unsigned long long orbitcta_forward_storage[];
    P10DC_ORBITCTA_CTX& c = *reinterpret_cast<P10DC_ORBITCTA_CTX*>(orbitcta_forward_storage);

    if (threadIdx.x == 0) {
        const uint32_t na = D_BKF_HIGH_NN_OFF[oi], nb = D_BKF_HIGH_NN_OFF[oi + 1];
        const uint32_t ra = D_BKF_HIGH_NRNL_OFF[oi], rb = D_BKF_HIGH_NRNL_OFF[oi + 1];
        c.n0 = nb - na;
        c.n1 = ra;
        c.total = c.n0 + (rb - ra);
    }
    __syncthreads();
    const uint32_t total = c.total;

    for (uint32_t k = uint32_t(blockIdx.y); k < total; k += uint32_t(gridDim.y)) {
        if (threadIdx.x == 0) {
            c.valid = 0;
            const bool nn = k < c.n0;
            const uint32_t qi = nn ? D_BKF_HIGH_NN_OFF[oi] + k : c.n1 + k - c.n0;
            const uint32_t sid = nn ? 0u : 3u;
            const BucketOrbitOp op = nn ? D_BKF_HIGH_NN[qi] : D_BKF_HIGH_NRNL[qi];
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

        if (c.valid) {
            constexpr int ILP = P10DC_ORBITCTA_COL_ILP;
            const BucketPhysicalBlock xb = c.xb;
            const BucketPhysicalBlock db = c.db;
            Count* const ip_base = c.ip_base;
            Count* const jp_base = c.jp_base;
            Count* const dp_base = c.dp_base;
            const uint32_t kind = c.kind;
            const uint32_t lane_step = uint32_t(blockDim.x);
            const uint32_t group_step = lane_step * uint32_t(ILP);
            for (uint32_t base = uint32_t(threadIdx.x); base < xb.cols; base += group_step) {
                uint32_t lr[ILP]{};
                uint8_t valid[ILP]{};
                Count x[ILP]{}, old[ILP]{}, y[ILP]{}, extra[ILP]{};
#pragma unroll
                for (int j = 0; j < ILP; ++j) {
                    lr[j] = base + uint32_t(j) * lane_step;
                    valid[j] = uint8_t(lr[j] < xb.cols);
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
                            P10DC_ORBITCTA_PLAN_SUM_PAIR(c, db, lr[j], lr[j + 1], extra[j], extra[j + 1]);
                        } else {
                            if (valid[j]) extra[j] = P10DC_ORBITCTA_PLAN_SUM(c, db, lr[j]);
                            if (valid[j + 1]) extra[j + 1] = P10DC_ORBITCTA_PLAN_SUM(c, db, lr[j + 1]);
                        }
                    } else if (valid[j]) {
                        extra[j] = P10DC_ORBITCTA_PLAN_SUM(c, db, lr[j]);
                    }
                }
#else
#pragma unroll
                for (int j = 0; j < ILP; ++j)
                    if (valid[j]) extra[j] = P10DC_ORBITCTA_PLAN_SUM(c, db, lr[j]);
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
        __syncthreads();
    }
}

__global__ void bucket_reverse_high_pattern10_depthcode_orbitcta_kernel(int p) {
    if (blockIdx.x) return;
    const uint32_t bid = blockIdx.z;
    if (bid >= D_BKF_MAIN_NBLOCKS) return;
    const uint32_t pi = uint32_t(p - (LOW_LUT_K + 1));
    const uint32_t oi = uint32_t(size_t(pi) * D_RS54_PITCH + bid);
    const bool edge = p == TARGET_W - 1;
    extern __shared__ unsigned long long orbitcta_reverse_storage[];
    P10DC_ORBITCTA_CTX& c = *reinterpret_cast<P10DC_ORBITCTA_CTX*>(orbitcta_reverse_storage);

    if (threadIdx.x == 0) {
        const uint32_t na = D_RS54_HIGH_NN_OFF[oi], nb = D_RS54_HIGH_NN_OFF[oi + 1];
        const uint32_t ra = D_RS54_HIGH_NR_OFF[oi], rb = D_RS54_HIGH_NR_OFF[oi + 1];
        const uint32_t la = D_RS54_HIGH_NL_OFF[oi], lb = D_RS54_HIGH_NL_OFF[oi + 1];
        c.n0 = nb - na;
        c.n1 = rb - ra;
        c.total = c.n0 + c.n1 + (lb - la);
    }
    __syncthreads();
    const uint32_t total = c.total;

    for (uint32_t k = uint32_t(blockIdx.y); k < total; k += uint32_t(gridDim.y)) {
        if (threadIdx.x == 0) {
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

        if (c.valid) {
            constexpr int ILP = P10DC_ORBITCTA_COL_ILP;
            const BucketPhysicalBlock xb = c.xb;
            const BucketPhysicalBlock db = c.db;
            const BucketPhysicalBlock sum_db = edge ? xb : db;
            Count* const ip_base = c.ip_base;
            Count* const jp_base = c.jp_base;
            Count* const dp_base = c.dp_base;
            const uint32_t kind = c.kind;
            const uint32_t lane_step = uint32_t(blockDim.x);
            const uint32_t group_step = lane_step * uint32_t(ILP);
            for (uint32_t base = uint32_t(threadIdx.x); base < xb.cols; base += group_step) {
                uint32_t lr[ILP]{};
                uint8_t valid[ILP]{};
                Count x[ILP]{}, old[ILP]{}, y[ILP]{}, extra[ILP]{};
#pragma unroll
                for (int j = 0; j < ILP; ++j) {
                    lr[j] = base + uint32_t(j) * lane_step;
                    valid[j] = uint8_t(lr[j] < xb.cols);
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
