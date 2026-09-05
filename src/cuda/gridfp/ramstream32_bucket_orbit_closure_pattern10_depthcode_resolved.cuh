#pragma once

#include "ramstream32_bucket_orbit_closure_pattern10_depthcode.cuh"

struct P10DCHighResolvedCtx {
    BucketPhysicalBlock xb{}, jb{}, db{};
    BkczPlan plan{};
    Count* ip_base = nullptr;
    Count* jp_base = nullptr;
    Count* dp_base = nullptr;
    Count* local_base[BKCZ_MAX_LOCAL]{};
    Count* cross_base = nullptr;
    uint32_t n0 = 0, n1 = 0, total = 0;
    uint8_t local_n = 0, kind = 0, valid = 0, pad = 0;
};

__device__ __forceinline__ void p10dc_resolve_high_rows(
    P10DCHighResolvedCtx& c,
    uint32_t ss, uint32_t js, uint32_t ds,
    uint32_t sr, uint32_t jr, uint32_t dr
) {
    c.ip_base = bkf_ptr(ss, c.xb.off + Code(sr) * c.xb.cols);
    c.jp_base = bkf_ptr(js, c.jb.off + Code(jr) * c.jb.cols);
    c.dp_base = bkf_ptr(ds, c.db.off + Code(dr) * c.db.cols);

    uint32_t n = bkcz_plan_local_n(c.plan);
    c.local_n = uint8_t(n);
#pragma unroll
    for (uint32_t i = 0; i < BKCZ_MAX_LOCAL; ++i) {
        if (i < n) {
            uint32_t x = c.plan.local[i];
            uint32_t sl = bkf_src_locator(x), owner = bkf_loc_owner(sl);
            BucketPhysicalBlock sb = bkf_high_main(owner, bkf_src_block(x));
            c.local_base[i] = bkf_ptr(
                owner, sb.off + Code(bkf_loc_rank(sl)) * sb.cols);
        }
    }

    c.cross_base = nullptr;
    if (bkcz_plan_cross_depth(c.plan)) {
        uint32_t x = bkcz_plan_cross_src(c.plan);
        uint32_t sl = bkf_src_locator(x), owner = bkf_loc_owner(sl);
        BucketPhysicalBlock sb = bkf_high_main(owner, bkf_src_block(x));
        c.cross_base = bkf_ptr(
            owner, sb.off + Code(bkf_loc_rank(sl)) * sb.cols);
    }
}

__device__ __forceinline__ BkczCrossAccum p10dc_resolved_low_preimages(
    uint32_t dest_code, uint32_t depth, const Count* source_row
) {
    BkczCrossAccum sum = 0;
    int s = int(depth);
    uint32_t key = bkcz_ternary_key<LOW_LUT_K>(dest_code);
    uint32_t weight = bkcz_pow3_const(LOW_LUT_K - 1);
#pragma unroll
    for (int pos = LOW_LUT_K - 1; pos >= 0; --pos) {
        uint32_t v = (dest_code >> (2 * pos)) & 3u;
        if (v == uint32_t(R)) {
            if (s == 1) break;
            --s;
        } else if (v == uint32_t(::L)) {
            if (s == 1) {
                uint32_t x = D_BKF_LOW_DIRECT[key - weight];
                if (x != BKF_DIRECT_INVALID)
                    sum = bkcz_cross_add(sum, source_row[bkf_loc_rank(x)]);
            }
            ++s;
        }
        if (pos) weight /= 3u;
    }
    return sum;
}

__device__ __forceinline__ Count p10dc_resolved_high_plan_sum(
    const P10DCHighResolvedCtx& c,
    const BucketPhysicalBlock& db,
    uint32_t lr
) {
#if GPU_DIRECT_PM_ACCUM
    uint64_t sum = 0;
#else
    Count sum = 0;
#endif
#pragma unroll
    for (uint32_t i = 0; i < BKCZ_MAX_LOCAL; ++i) {
        if (i < c.local_n) {
            Count v = c.local_base[i][lr];
#if GPU_DIRECT_PM_ACCUM
            sum += uint64_t(v);
#else
            sum = gpu_direct_add(sum, v);
#endif
        }
    }
    uint32_t depth = bkcz_plan_cross_depth(c.plan);
    if (depth) {
        uint32_t dc = D_BKF_LOW_CODES[
            D_BKF_LOW_CODE_OFF[size_t(D_BKF_FIXED_OWNER) * D_BKF_CODE_PITCH + db.hs] + lr];
#if GPU_DIRECT_PM_ACCUM
        sum += p10dc_resolved_low_preimages(dc, depth, c.cross_base);
#else
        sum = gpu_direct_add(
            sum, p10dc_resolved_low_preimages(dc, depth, c.cross_base));
#endif
    }
#if GPU_DIRECT_PM_ACCUM
    return gpu_direct_pm_reduce_u64(sum);
#else
    return sum;
#endif
}

__global__ void bucket_high_orbit_closure_pattern10_depthcode_resolved_kernel(int p) {
    uint32_t bid = blockIdx.z;
    if (bid >= D_BKF_MAIN_NBLOCKS) return;
    uint32_t pi = uint32_t((TARGET_W - 1) - p);
    uint32_t oi = uint32_t(size_t(pi) * D_BKF_HIGH_PITCH + bid);
    __shared__ P10DCHighResolvedCtx c;

    if (threadIdx.x == 0) {
        uint32_t na = D_BKF_HIGH_NN_OFF[oi], nb = D_BKF_HIGH_NN_OFF[oi + 1];
        uint32_t ra = D_BKF_HIGH_NRNL_OFF[oi], rb = D_BKF_HIGH_NRNL_OFF[oi + 1];
        c.n0 = nb - na;
        c.n1 = ra;
        c.total = c.n0 + (rb - ra);
    }
    __syncthreads();

    for (uint32_t k = blockIdx.y; k < c.total; k += gridDim.y) {
        if (threadIdx.x == 0) {
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
                uint32_t payload = p10dc_payload(
                    op, false, true, sid, p, uint32_t(c.xb.hs));
                c.plan = p10dc_forward_high(payload, dl, c.db, p);
                c.kind = uint8_t(nn ? CPU_ORBIT_NN : CPU_ORBIT_NR);
                p10dc_resolve_high_rows(
                    c, ss, js, ds,
                    bkf_loc_rank(sl), bkf_loc_rank(jl), bkf_loc_rank(dl));
                c.valid = 1;
            }
        }
        __syncthreads();

        if (c.valid) {
            for (uint32_t lr = uint32_t(blockIdx.x) * blockDim.x + threadIdx.x;
                 lr < c.xb.cols; lr += uint32_t(gridDim.x) * blockDim.x) {
                Count* ip = c.ip_base + lr;
                Count* jp = c.jp_base + lr;
                Count* dp = c.dp_base + lr;
                Count x = *ip, old = *dp;
                Count extra = p10dc_resolved_high_plan_sum(c, c.db, lr);
                if (c.kind == CPU_ORBIT_NN) {
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
        __syncthreads();
    }
}

__global__ void bucket_reverse_high_pattern10_depthcode_resolved_kernel(int p) {
    uint32_t bid = blockIdx.z;
    if (bid >= D_BKF_MAIN_NBLOCKS) return;
    uint32_t pi = uint32_t(p - (LOW_LUT_K + 1));
    uint32_t oi = uint32_t(size_t(pi) * D_RS54_PITCH + bid);
    bool edge = p == TARGET_W - 1;
    __shared__ P10DCHighResolvedCtx c;

    if (threadIdx.x == 0) {
        uint32_t na = D_RS54_HIGH_NN_OFF[oi], nb = D_RS54_HIGH_NN_OFF[oi + 1];
        uint32_t ra = D_RS54_HIGH_NR_OFF[oi], rb = D_RS54_HIGH_NR_OFF[oi + 1];
        uint32_t la = D_RS54_HIGH_NL_OFF[oi], lb = D_RS54_HIGH_NL_OFF[oi + 1];
        c.n0 = nb - na;
        c.n1 = rb - ra;
        c.total = c.n0 + c.n1 + (lb - la);
    }
    __syncthreads();

    for (uint32_t k = blockIdx.y; k < c.total; k += gridDim.y) {
        if (threadIdx.x == 0) {
            c.valid = 0;
            uint32_t qi = 0, kind = 0, sid = 0;
            BucketOrbitOp op;
            if (k < c.n0) {
                kind = CPU_ORBIT_NN; sid = 0;
                qi = D_RS54_HIGH_NN_OFF[oi] + k; op = D_RS54_HIGH_NN[qi];
            } else if (k < c.n0 + c.n1) {
                kind = CPU_ORBIT_NR; sid = 1;
                qi = D_RS54_HIGH_NR_OFF[oi] + k - c.n0; op = D_RS54_HIGH_NR[qi];
            } else {
                kind = CPU_ORBIT_NL; sid = 2;
                qi = D_RS54_HIGH_NL_OFF[oi] + k - c.n0 - c.n1; op = D_RS54_HIGH_NL[qi];
            }
            uint32_t sl = bkf_orbit_src(op), jl = bkf_orbit_partner(op), dl = bkf_orbit_drop(op);
            uint32_t ss = bkf_loc_owner(sl), js = bkf_loc_owner(jl), ds = bkf_loc_owner(dl);
            c.xb = bkf_high_main(ss, bid);
            if (c.xb.valid && c.xb.rows && c.xb.cols) {
                c.jb = bkf_high_main(js, bkcp10_reverse_high_jblock(bid, c.xb, p, kind));
                c.db = bkf_high_block(ds, uint32_t(c.xb.hs));
                uint32_t payload = p10dc_payload(
                    op, true, true, sid, p, uint32_t(c.xb.hs));
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
        __syncthreads();

        if (c.valid) {
            for (uint32_t lr = uint32_t(blockIdx.x) * blockDim.x + threadIdx.x;
                 lr < c.xb.cols; lr += uint32_t(gridDim.x) * blockDim.x) {
                Count* ip = c.ip_base + lr;
                Count* jp = c.jp_base + lr;
                Count* dp = c.dp_base + lr;
                Count x = *ip, old = *dp;
                Count extra = p10dc_resolved_high_plan_sum(c, edge ? c.xb : c.db, lr);
                if (c.kind == CPU_ORBIT_NN) {
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
        __syncthreads();
    }
}
