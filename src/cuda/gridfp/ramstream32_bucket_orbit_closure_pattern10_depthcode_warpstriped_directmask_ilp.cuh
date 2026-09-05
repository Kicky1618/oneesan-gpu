#pragma once

#include "ramstream32_bucket_orbit_closure_pattern10_depthcode_warpctx.cuh"
#include "ramstream32_bucket_closure_pattern10_depthcode_delta_direct_affine_rankchunk32_directmask_ilp.cuh"

static_assert(P10DC_RANKCHUNK32_DIRECTMASK,
              "directmask ILP kernel requires P10DC_RANKCHUNK32_DIRECTMASK=1");
static_assert(P10DC_WARPSTRIPED_ILP == 2 || P10DC_WARPSTRIPED_ILP == 4,
              "specialized directmask ILP kernel requires ILP=2 or 4");

#ifndef P10DC_WARPSTRIPED_THREADS_OK_DEFINED
#define P10DC_WARPSTRIPED_THREADS_OK_DEFINED 1
static inline bool p10dc_warpstriped_threads_ok(int threads) {
    return threads > 0 && threads <= 1024 && (threads & 31) == 0;
}
#endif

__global__ void
bucket_high_orbit_closure_pattern10_depthcode_warpstriped_delta_direct_affine_rankchunk32_cross5_kernel(
    int p
) {
    uint32_t bid = blockIdx.z;
    if (bid >= D_BKF_MAIN_NBLOCKS) return;
    uint32_t pi = uint32_t((TARGET_W - 1) - p);
    uint32_t oi = uint32_t(size_t(pi) * D_BKF_HIGH_PITCH + bid);
    extern __shared__ unsigned long long warp_ctx_storage[];
    auto* warp_ctx = reinterpret_cast<P10DCDirectHighResolvedCtx*>(warp_ctx_storage);

    const uint32_t lane = uint32_t(threadIdx.x) & 31u;
    const uint32_t warp = uint32_t(threadIdx.x) >> 5;
    const uint32_t nwarps = uint32_t(blockDim.x) >> 5;
    const unsigned active = __activemask();
    P10DCDirectHighResolvedCtx& c = warp_ctx[warp];

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
                const uint32_t payload =
                    p10dc_payload(op, false, true, sid, p, uint32_t(c.xb.hs));
                p10dc_prepare_forward_high_delta_direct_affine(
                    c, payload, dl, p, ss, js, ds,
                    bkf_loc_rank(sl), bkf_loc_rank(jl), bkf_loc_rank(dl));
                c.kind = uint8_t(nn ? CPU_ORBIT_NN : CPU_ORBIT_NR);
                c.valid = 1;
            }
        }
        __syncwarp(active);

        if (c.valid) {
            constexpr int ILP = P10DC_WARPSTRIPED_ILP;
            const BucketPhysicalBlock xb = c.xb;
            const BucketPhysicalBlock db = c.db;
            Count* const ip_base = c.ip_base;
            Count* const jp_base = c.jp_base;
            Count* const dp_base = c.dp_base;
            const uint32_t kind = c.kind;
            const uint32_t stripe = uint32_t(gridDim.x) * 32u;
            const uint32_t group_stride = stripe * uint32_t(ILP);

            for (uint32_t base = uint32_t(blockIdx.x) * 32u + lane;
                 base < xb.cols; base += group_stride) {
                uint32_t lr[ILP]{};
                uint8_t valid[ILP]{};
                Count x[ILP]{}, old[ILP]{}, y[ILP]{}, extra[ILP]{};

#pragma unroll
                for (int j = 0; j < ILP; ++j) {
                    lr[j] = base + uint32_t(j) * stripe;
                    valid[j] = lr[j] < xb.cols;
                    if (valid[j]) {
                        x[j] = ip_base[lr[j]];
                        old[j] = dp_base[lr[j]];
                        y[j] = jp_base[lr[j]];
                    }
                }

                p10dc_direct_resolved_high_plan_sum_rankchunk32_directmask_ilp<ILP>(
                    c, db, lr, valid, extra);

#pragma unroll
                for (int j = 0; j < ILP; ++j) {
                    if (!valid[j]) continue;
                    if (kind == CPU_ORBIT_NN) {
                        jp_base[lr[j]] = gpu_direct_add(y[j], x[j]);
                        ip_base[lr[j]] = gpu_direct_add(x[j], old[j]);
                        dp_base[lr[j]] = extra[j];
                    } else {
                        ip_base[lr[j]] = gpu_direct_add(gpu_direct_add(x[j], y[j]), old[j]);
                        dp_base[lr[j]] = gpu_direct_add(x[j], extra[j]);
                    }
                }
            }
        }
        __syncwarp(active);
    }
}

__global__ void
bucket_reverse_high_pattern10_depthcode_warpstriped_delta_direct_affine_rankchunk32_cross5_kernel(
    int p
) {
    uint32_t bid = blockIdx.z;
    if (bid >= D_BKF_MAIN_NBLOCKS) return;
    uint32_t pi = uint32_t(p - (LOW_LUT_K + 1));
    uint32_t oi = uint32_t(size_t(pi) * D_RS54_PITCH + bid);
    const bool edge = p == TARGET_W - 1;
    extern __shared__ unsigned long long warp_ctx_storage[];
    auto* warp_ctx = reinterpret_cast<P10DCDirectHighResolvedCtx*>(warp_ctx_storage);

    const uint32_t lane = uint32_t(threadIdx.x) & 31u;
    const uint32_t warp = uint32_t(threadIdx.x) >> 5;
    const uint32_t nwarps = uint32_t(blockDim.x) >> 5;
    const unsigned active = __activemask();
    P10DCDirectHighResolvedCtx& c = warp_ctx[warp];

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
            const uint32_t sl = bkf_orbit_src(op), jl = bkf_orbit_partner(op), dl = bkf_orbit_drop(op);
            const uint32_t ss = bkf_loc_owner(sl), js = bkf_loc_owner(jl), ds = bkf_loc_owner(dl);
            c.xb = bkf_high_main(ss, bid);
            if (c.xb.valid && c.xb.rows && c.xb.cols) {
                c.jb = bkf_high_main(js, bkcp10_reverse_high_jblock(bid, c.xb, p, kind));
                c.db = bkf_high_block(ds, uint32_t(c.xb.hs));
                const uint32_t payload =
                    p10dc_payload(op, true, true, sid, p, uint32_t(c.xb.hs));
                p10dc_prepare_reverse_high_delta_direct_affine(
                    c, payload, edge ? sl : dl, edge ? c.xb : c.db, p, edge,
                    ss, js, ds, bkf_loc_rank(sl), bkf_loc_rank(jl), bkf_loc_rank(dl));
                c.kind = uint8_t(kind);
                c.valid = 1;
            }
        }
        __syncwarp(active);

        if (c.valid) {
            constexpr int ILP = P10DC_WARPSTRIPED_ILP;
            const BucketPhysicalBlock xb = c.xb;
            const BucketPhysicalBlock db = c.db;
            const BucketPhysicalBlock sum_db = edge ? xb : db;
            Count* const ip_base = c.ip_base;
            Count* const jp_base = c.jp_base;
            Count* const dp_base = c.dp_base;
            const uint32_t kind = c.kind;
            const uint32_t stripe = uint32_t(gridDim.x) * 32u;
            const uint32_t group_stride = stripe * uint32_t(ILP);

            for (uint32_t base = uint32_t(blockIdx.x) * 32u + lane;
                 base < xb.cols; base += group_stride) {
                uint32_t lr[ILP]{};
                uint8_t valid[ILP]{};
                Count x[ILP]{}, old[ILP]{}, y[ILP]{}, extra[ILP]{};

#pragma unroll
                for (int j = 0; j < ILP; ++j) {
                    lr[j] = base + uint32_t(j) * stripe;
                    valid[j] = lr[j] < xb.cols;
                    if (valid[j]) {
                        x[j] = ip_base[lr[j]];
                        old[j] = dp_base[lr[j]];
                        y[j] = jp_base[lr[j]];
                    }
                }

                p10dc_direct_resolved_high_plan_sum_rankchunk32_directmask_ilp<ILP>(
                    c, sum_db, lr, valid, extra);

#pragma unroll
                for (int j = 0; j < ILP; ++j) {
                    if (!valid[j]) continue;
                    if (kind == CPU_ORBIT_NN) {
                        jp_base[lr[j]] = gpu_direct_add(y[j], x[j]);
                        ip_base[lr[j]] = gpu_direct_add(
                            gpu_direct_add(x[j], old[j]), edge ? extra[j] : 0);
                        dp_base[lr[j]] = edge ? 0 : extra[j];
                    } else {
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
        __syncwarp(active);
    }
}
