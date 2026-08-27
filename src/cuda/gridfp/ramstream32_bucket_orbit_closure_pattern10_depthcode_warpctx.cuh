#pragma once

#include "ramstream32_bucket_orbit_closure_pattern10_depthcode_resolved.cuh"

// Warp-private variant of the resolved HIGH context. Lane 0 resolves the orbit,
// destination rows and closure-source rows once per warp. The warp then consumes
// a private shared-memory slot, replacing block-wide barriers with __syncwarp().
// This intentionally keeps the same logical column mapping as the 256-thread
// kernels: all warps process the same orbit, but different threadIdx.x columns.
static constexpr int P10DC_WARPCTX_MAX_WARPS = 32;

__global__ void bucket_high_orbit_closure_pattern10_depthcode_warpctx_kernel(int p) {
    uint32_t bid = blockIdx.z;
    if (bid >= D_BKF_MAIN_NBLOCKS) return;
    uint32_t pi = uint32_t((TARGET_W - 1) - p);
    uint32_t oi = uint32_t(size_t(pi) * D_BKF_HIGH_PITCH + bid);
    __shared__ P10DCHighResolvedCtx warp_ctx[P10DC_WARPCTX_MAX_WARPS];

    const uint32_t lane = uint32_t(threadIdx.x) & 31u;
    const uint32_t warp = uint32_t(threadIdx.x) >> 5;
    const unsigned active = __activemask();
    P10DCHighResolvedCtx& c = warp_ctx[warp];

    if (lane == 0) {
        uint32_t na = D_BKF_HIGH_NN_OFF[oi], nb = D_BKF_HIGH_NN_OFF[oi + 1];
        uint32_t ra = D_BKF_HIGH_NRNL_OFF[oi], rb = D_BKF_HIGH_NRNL_OFF[oi + 1];
        c.n0 = nb - na;
        c.n1 = ra;
        c.total = c.n0 + (rb - ra);
    }
    __syncwarp(active);
    const uint32_t total = c.total;

    for (uint32_t k = blockIdx.y; k < total; k += gridDim.y) {
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
                c.plan = p10dc_forward_high(payload, dl, c.db, p);
                c.kind = uint8_t(nn ? CPU_ORBIT_NN : CPU_ORBIT_NR);
                p10dc_resolve_high_rows(
                    c, ss, js, ds,
                    bkf_loc_rank(sl), bkf_loc_rank(jl), bkf_loc_rank(dl));
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
            for (uint32_t lr = uint32_t(blockIdx.x) * blockDim.x + threadIdx.x;
                 lr < xb.cols;
                 lr += uint32_t(gridDim.x) * blockDim.x) {
                Count* ip = ip_base + lr;
                Count* jp = jp_base + lr;
                Count* dp = dp_base + lr;
                Count x = *ip, old = *dp;
                Count extra = p10dc_resolved_high_plan_sum(c, db, lr);
                if (kind == CPU_ORBIT_NN) {
                    *jp = gpu_direct_add(*jp, x);
                    *ip = gpu_direct_add(x, old);
                    *dp = extra;
                } else {
                    Count y = *jp;
                    *ip = gpu_direct_add(gpu_direct_add(x, y), old);
                    *dp = gpu_direct_add(x, extra);
                }
            }
        }
        __syncwarp(active);
    }
}

__global__ void bucket_reverse_high_pattern10_depthcode_warpctx_kernel(int p) {
    uint32_t bid = blockIdx.z;
    if (bid >= D_BKF_MAIN_NBLOCKS) return;
    uint32_t pi = uint32_t(p - (LOW_LUT_K + 1));
    uint32_t oi = uint32_t(size_t(pi) * D_RS54_PITCH + bid);
    const bool edge = p == TARGET_W - 1;
    __shared__ P10DCHighResolvedCtx warp_ctx[P10DC_WARPCTX_MAX_WARPS];

    const uint32_t lane = uint32_t(threadIdx.x) & 31u;
    const uint32_t warp = uint32_t(threadIdx.x) >> 5;
    const unsigned active = __activemask();
    P10DCHighResolvedCtx& c = warp_ctx[warp];

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

    for (uint32_t k = blockIdx.y; k < total; k += gridDim.y) {
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
                c.plan = edge
                    ? p10dc_reverse_high(payload, sl, c.xb, p, true)
                    : p10dc_reverse_high(payload, dl, c.db, p, false);
                c.kind = uint8_t(kind);
                p10dc_resolve_high_rows(
                    c, ss, js, ds,
                    bkf_loc_rank(sl), bkf_loc_rank(jl), bkf_loc_rank(dl));
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
            for (uint32_t lr = uint32_t(blockIdx.x) * blockDim.x + threadIdx.x;
                 lr < xb.cols;
                 lr += uint32_t(gridDim.x) * blockDim.x) {
                Count* ip = ip_base + lr;
                Count* jp = jp_base + lr;
                Count* dp = dp_base + lr;
                Count x = *ip, old = *dp;
                Count extra = p10dc_resolved_high_plan_sum(c, edge ? xb : db, lr);
                if (kind == CPU_ORBIT_NN) {
                    *jp = gpu_direct_add(*jp, x);
                    *ip = gpu_direct_add(gpu_direct_add(x, old), edge ? extra : 0);
                    *dp = edge ? 0 : extra;
                } else {
                    Count y = *jp;
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
        __syncwarp(active);
    }
}
