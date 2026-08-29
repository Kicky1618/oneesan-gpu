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

// Bandwidth-oriented HIGH scheduler.  Unlike warpstriped X slicing, each CTA
// owns a whole orbit and builds its row/closure context once.  All block threads
// then walk consecutive LOW-rank columns.  Launch with grid.x=1; grid.y controls
// orbit concurrency.  This removes the gx-fold duplicated lane0 setup that
// becomes dominant after DIRECTGATHER makes each column cheap.
__global__ void bucket_high_orbit_closure_pattern10_depthcode_orbitcta_kernel(int p) {
    if (blockIdx.x) return;
    const uint32_t bid = blockIdx.z;
    if (bid >= D_BKF_MAIN_NBLOCKS) return;
    const uint32_t pi = uint32_t((TARGET_W - 1) - p);
    const uint32_t oi = uint32_t(size_t(pi) * D_BKF_HIGH_PITCH + bid);
    __shared__ P10DC_ORBITCTA_CTX c;

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
            const BucketPhysicalBlock xb = c.xb;
            const BucketPhysicalBlock db = c.db;
            Count* const ip_base = c.ip_base;
            Count* const jp_base = c.jp_base;
            Count* const dp_base = c.dp_base;
            const uint32_t kind = c.kind;
            for (uint32_t lr = uint32_t(threadIdx.x); lr < xb.cols; lr += uint32_t(blockDim.x)) {
                Count* const ip = ip_base + lr;
                Count* const jp = jp_base + lr;
                Count* const dp = dp_base + lr;
                const Count x = *ip;
                const Count old = *dp;
#if P10DC_ORBITCTA_EARLY_JP
                const Count y = *jp;
#endif
                const Count extra = P10DC_ORBITCTA_PLAN_SUM(c, db, lr);
                if (kind == CPU_ORBIT_NN) {
#if P10DC_ORBITCTA_EARLY_JP
                    *jp = gpu_direct_add(y, x);
#else
                    *jp = gpu_direct_add(*jp, x);
#endif
                    *ip = gpu_direct_add(x, old);
                    *dp = extra;
                } else {
#if !P10DC_ORBITCTA_EARLY_JP
                    const Count y = *jp;
#endif
                    *ip = gpu_direct_add(gpu_direct_add(x, y), old);
                    *dp = gpu_direct_add(x, extra);
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
    __shared__ P10DC_ORBITCTA_CTX c;

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
            const BucketPhysicalBlock xb = c.xb;
            const BucketPhysicalBlock db = c.db;
            Count* const ip_base = c.ip_base;
            Count* const jp_base = c.jp_base;
            Count* const dp_base = c.dp_base;
            const uint32_t kind = c.kind;
            for (uint32_t lr = uint32_t(threadIdx.x); lr < xb.cols; lr += uint32_t(blockDim.x)) {
                Count* const ip = ip_base + lr;
                Count* const jp = jp_base + lr;
                Count* const dp = dp_base + lr;
                const Count x = *ip;
                const Count old = *dp;
#if P10DC_ORBITCTA_EARLY_JP
                const Count y = *jp;
#endif
                const Count extra = P10DC_ORBITCTA_PLAN_SUM(c, edge ? xb : db, lr);
                if (kind == CPU_ORBIT_NN) {
#if P10DC_ORBITCTA_EARLY_JP
                    *jp = gpu_direct_add(y, x);
#else
                    *jp = gpu_direct_add(*jp, x);
#endif
                    *ip = gpu_direct_add(gpu_direct_add(x, old), edge ? extra : 0);
                    *dp = edge ? 0 : extra;
                } else {
#if !P10DC_ORBITCTA_EARLY_JP
                    const Count y = *jp;
#endif
                    *ip = gpu_direct_add(gpu_direct_add(x, y), old);
                    if (edge) {
                        *jp = gpu_direct_add(x, y);
                        *dp = 0;
                    } else {
                        *dp = gpu_direct_add(x, extra);
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
