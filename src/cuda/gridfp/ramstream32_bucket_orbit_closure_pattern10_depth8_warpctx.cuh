#pragma once

#include "ramstream32_bucket_orbit_closure_pattern10_depth8_highctx.cuh"

// Middle ground between the original per-thread HIGH setup and the block-wide
// shared-context kernel: lane 0 of each warp builds one context, then the warp
// consumes it from a private shared-memory slot.  This duplicates setup once
// per warp instead of once per thread, while replacing the two block-wide
// __syncthreads() barriers per orbit operation with warp-local barriers.
//
// CUDA blocks are limited to 1024 threads, so 32 warp slots cover every legal
// launch shape.  Partial final warps are handled with __activemask().
static constexpr int P10D8_WARPCTX_MAX_WARPS = 32;

__global__ void bucket_high_orbit_closure_pattern10_depth8_warpctx_kernel(int p) {
    uint32_t bid = blockIdx.z;
    if (bid >= D_BKF_MAIN_NBLOCKS) return;

    uint32_t pi = uint32_t((TARGET_W - 1) - p);
    uint32_t oi = uint32_t(size_t(pi) * D_BKF_HIGH_PITCH + bid);
    __shared__ P10D8HighCtx warp_ctx[P10D8_WARPCTX_MAX_WARPS];

    const uint32_t lane = uint32_t(threadIdx.x) & 31u;
    const uint32_t warp = uint32_t(threadIdx.x) >> 5;
    const unsigned active = __activemask();
    P10D8HighCtx& c = warp_ctx[warp];

    if (lane == 0) {
        uint32_t na = D_BKF_HIGH_NN_OFF[oi];
        uint32_t nb = D_BKF_HIGH_NN_OFF[oi + 1];
        uint32_t ra = D_BKF_HIGH_NRNL_OFF[oi];
        uint32_t rb = D_BKF_HIGH_NRNL_OFF[oi + 1];
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
            uint32_t qi = nn ? (D_BKF_HIGH_NN_OFF[oi] + k) : (c.n1 + k - c.n0);
            BucketOrbitOp op = nn ? D_BKF_HIGH_NN[qi] : D_BKF_HIGH_NRNL[qi];
            uint8_t dep = nn ? D_P10D8_F_HIGH_NN[qi] : D_P10D8_F_HIGH_NRNL[qi];
            uint32_t sl = bkf_orbit_src(op);
            uint32_t jl = bkf_orbit_partner(op);
            uint32_t dl = bkf_orbit_drop(op);
            c.ss = bkf_loc_owner(sl);
            c.js = bkf_loc_owner(jl);
            c.ds = bkf_loc_owner(dl);
            c.xb = bkf_high_main(c.ss, bid);

            if (c.xb.valid && c.xb.rows && c.xb.cols) {
                uint32_t jbid = bid;
                if (p == LOW_LUT_K + 1) {
                    uint32_t center = nn ? uint32_t(R) : uint32_t(N);
                    int he = int(c.xb.hs) + (center == uint32_t(R) ? 1 : 0);
                    jbid = uint32_t(3 * he + int(center));
                }
                c.jb = bkf_high_main(c.js, jbid);
                c.db = bkf_high_block(c.ds, uint32_t(c.xb.hs));
                c.sr = bkf_loc_rank(sl);
                c.jr = bkf_loc_rank(jl);
                c.dr = bkf_loc_rank(dl);
                c.kind = uint8_t(nn ? CPU_ORBIT_NN : CPU_ORBIT_NR);
                c.plan = bkcpd8_forward_high(bkcp10_id(op), dep, dl, c.db, p);
                c.valid = 1;
            }
        }
        __syncwarp(active);

        if (c.valid) {
            // Keep the addressing context in registers like the original
            // per-thread kernel; only the closure plan remains warp-shared.
            const BucketPhysicalBlock xb = c.xb;
            const BucketPhysicalBlock jb = c.jb;
            const BucketPhysicalBlock db = c.db;
            const uint32_t ss = c.ss, js = c.js, ds = c.ds;
            const uint32_t sr = c.sr, jr = c.jr, dr = c.dr;
            const uint32_t kind = c.kind;
            const BkczPlan& plan = c.plan;

            for (uint32_t lr = uint32_t(blockIdx.x) * blockDim.x + threadIdx.x;
                 lr < xb.cols;
                 lr += uint32_t(gridDim.x) * blockDim.x) {
                Count* ip = bkf_ptr(ss, xb.off + Code(sr) * xb.cols + lr);
                Count* jp = bkf_ptr(js, jb.off + Code(jr) * jb.cols + lr);
                Count* dp = bkf_ptr(ds, db.off + Code(dr) * db.cols + lr);
                Count x = *ip;
                Count old = *dp;
                Count extra = bkcz_high_plan_sum(plan, db, lr);
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

__global__ void bucket_reverse_high_pattern10_depth8_warpctx_kernel(int p) {
    uint32_t bid = blockIdx.z;
    if (bid >= D_BKF_MAIN_NBLOCKS) return;

    uint32_t pi = uint32_t(p - (LOW_LUT_K + 1));
    uint32_t oi = uint32_t(size_t(pi) * D_RS54_PITCH + bid);
    const bool edge = p == TARGET_W - 1;
    __shared__ P10D8HighCtx warp_ctx[P10D8_WARPCTX_MAX_WARPS];

    const uint32_t lane = uint32_t(threadIdx.x) & 31u;
    const uint32_t warp = uint32_t(threadIdx.x) >> 5;
    const unsigned active = __activemask();
    P10D8HighCtx& c = warp_ctx[warp];

    if (lane == 0) {
        uint32_t na = D_RS54_HIGH_NN_OFF[oi];
        uint32_t nb = D_RS54_HIGH_NN_OFF[oi + 1];
        uint32_t ra = D_RS54_HIGH_NR_OFF[oi];
        uint32_t rb = D_RS54_HIGH_NR_OFF[oi + 1];
        uint32_t la = D_RS54_HIGH_NL_OFF[oi];
        uint32_t lb = D_RS54_HIGH_NL_OFF[oi + 1];
        c.n0 = nb - na;
        c.n1 = rb - ra;
        c.total = c.n0 + c.n1 + (lb - la);
    }
    __syncwarp(active);
    const uint32_t total = c.total;

    for (uint32_t k = blockIdx.y; k < total; k += gridDim.y) {
        if (lane == 0) {
            c.valid = 0;
            uint32_t qi = 0;
            uint32_t kind = 0;
            BucketOrbitOp op;
            uint8_t dep;
            if (k < c.n0) {
                kind = CPU_ORBIT_NN;
                qi = D_RS54_HIGH_NN_OFF[oi] + k;
                op = D_RS54_HIGH_NN[qi];
                dep = D_P10D8_R_HIGH_NN[qi];
            } else if (k < c.n0 + c.n1) {
                kind = CPU_ORBIT_NR;
                qi = D_RS54_HIGH_NR_OFF[oi] + k - c.n0;
                op = D_RS54_HIGH_NR[qi];
                dep = D_P10D8_R_HIGH_NR[qi];
            } else {
                kind = CPU_ORBIT_NL;
                qi = D_RS54_HIGH_NL_OFF[oi] + k - c.n0 - c.n1;
                op = D_RS54_HIGH_NL[qi];
                dep = D_P10D8_R_HIGH_NL[qi];
            }

            uint32_t sl = bkf_orbit_src(op);
            uint32_t jl = bkf_orbit_partner(op);
            uint32_t dl = bkf_orbit_drop(op);
            c.ss = bkf_loc_owner(sl);
            c.js = bkf_loc_owner(jl);
            c.ds = bkf_loc_owner(dl);
            c.xb = bkf_high_main(c.ss, bid);

            if (c.xb.valid && c.xb.rows && c.xb.cols) {
                c.jb = bkf_high_main(c.js, bkcp10_reverse_high_jblock(bid, c.xb, p, kind));
                c.db = bkf_high_block(c.ds, uint32_t(c.xb.hs));
                c.sr = bkf_loc_rank(sl);
                c.jr = bkf_loc_rank(jl);
                c.dr = bkf_loc_rank(dl);
                c.kind = uint8_t(kind);
                c.plan = edge
                    ? bkcpd8_reverse_high(bkcp10_id(op), dep, sl, c.xb, p, true)
                    : bkcpd8_reverse_high(bkcp10_id(op), dep, dl, c.db, p, false);
                c.valid = 1;
            }
        }
        __syncwarp(active);

        if (c.valid) {
            const BucketPhysicalBlock xb = c.xb;
            const BucketPhysicalBlock jb = c.jb;
            const BucketPhysicalBlock db = c.db;
            const uint32_t ss = c.ss, js = c.js, ds = c.ds;
            const uint32_t sr = c.sr, jr = c.jr, dr = c.dr;
            const uint32_t kind = c.kind;
            const BkczPlan& plan = c.plan;

            for (uint32_t lr = uint32_t(blockIdx.x) * blockDim.x + threadIdx.x;
                 lr < xb.cols;
                 lr += uint32_t(gridDim.x) * blockDim.x) {
                Count* ip = bkf_ptr(ss, xb.off + Code(sr) * xb.cols + lr);
                Count* jp = bkf_ptr(js, jb.off + Code(jr) * jb.cols + lr);
                Count* dp = bkf_ptr(ds, db.off + Code(dr) * db.cols + lr);
                Count x = *ip;
                Count old = *dp;
                Count extra = bkcz_high_plan_sum(plan, edge ? xb : db, lr);
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

static void bucket_launch_high_orbit_closure_pattern10_depth8_warpctx(
    const StorageLayout& layout, int threads = 256, int gx = 16, int gy = 8
) {
    dim3 block(threads), grid(gx, gy, unsigned(layout.main_blocks.size()));
    for (int p = TARGET_W - 1; p >= LOW_LUT_K + 1; --p) {
        bucket_high_orbit_closure_pattern10_depth8_warpctx_kernel<<<grid, block>>>(p);
        ck(cudaGetLastError(), "bucket high pattern10 depth8 warpctx");
    }
}

static void bucket_launch_reverse_high_pattern10_depth8_warpctx(
    const StorageLayout& layout, int threads = 256, int gx = 16, int gy = 8
) {
    dim3 block(threads), grid(gx, gy, unsigned(layout.main_blocks.size()));
    for (int p = LOW_LUT_K + 1; p < TARGET_W; ++p) {
        bucket_reverse_high_pattern10_depth8_warpctx_kernel<<<grid, block>>>(p);
        ck(cudaGetLastError(), "bucket reverse high pattern10 depth8 warpctx");
    }
}
