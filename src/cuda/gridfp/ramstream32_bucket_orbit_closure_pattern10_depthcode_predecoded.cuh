#pragma once

#include "ramstream32_bucket_onepass_pattern10_depthcode_predecoded.hpp"
#include "ramstream32_bucket_closure_zero_plan.cuh"

__constant__ uint32_t* D_P10DCP_BASE;
__constant__ uint32_t* D_P10DCP_DECODE;
__constant__ uint32_t D_P10DCP_MODE;

struct P10DepthCodePredecodedDeviceTables {
    uint32_t* base = nullptr;
    uint32_t* decode = nullptr;
    template<class T>
    static void cp(T*& d, const std::vector<T>& s, const char* what) {
        if (s.empty()) return;
        ck(cudaMalloc(&d, s.size() * sizeof(T)), what);
        ck(cudaMemcpy(d, s.data(), s.size() * sizeof(T), cudaMemcpyHostToDevice), what);
    }
    void install(const P10DepthCodePredecodedBookHost& h) {
        cp(base, h.base, "p10dcp base");
        cp(decode, h.decode, "p10dcp decode");
        ck(cudaMemcpyToSymbol(D_P10DCP_BASE, &base, sizeof(base)), "p10dcp base ptr");
        ck(cudaMemcpyToSymbol(D_P10DCP_DECODE, &decode, sizeof(decode)), "p10dcp decode ptr");
        uint32_t mode = h.mode;
        ck(cudaMemcpyToSymbol(D_P10DCP_MODE, &mode, sizeof(mode)), "p10dcp mode");
    }
    void release() {
        cudaFree(base);
        cudaFree(decode);
        base = nullptr;
        decode = nullptr;
    }
};

struct BucketForwardPattern10DepthCodePredecodedDeviceTables {
    void install(const BucketForwardPattern10DepthCodePredecodedHost&) {}
    void release() {}
};
struct BucketReversePattern10DepthCodePredecodedDeviceTables {
    ReverseSplit54DeviceTables split;
    P10DepthCodePredecodedDeviceTables codebook;
    void install(const BucketReversePattern10DepthCodePredecodedHost& h) {
        split.install(h.split);
        codebook.install(h.codebook);
    }
    void release() {
        codebook.release();
        split.release();
    }
};

__device__ __forceinline__ uint32_t p10dcp_desc(
    BucketOrbitOp op, bool rev, bool high, uint32_t sid, int p, uint32_t h
) {
    uint32_t k = p10dc_key(rev, high, sid, p, h, D_P10DCP_MODE);
    uint32_t b = D_P10DCP_BASE[k];
    return D_P10DCP_DECODE[b + uint32_t(bkcp10_id(op))];
}
__device__ __forceinline__ bool p10dcp_active(uint32_t x) {
    return ((x >> P10DCP_ACTIVE_SHIFT) & 1u) != 0;
}
__device__ __forceinline__ uint16_t p10dcp_lm(uint32_t x) {
    return uint16_t(x & P10DCP_MASK_MASK);
}
__device__ __forceinline__ uint16_t p10dcp_rm(uint32_t x) {
    return uint16_t((x >> P10DCP_RM_SHIFT) & P10DCP_MASK_MASK);
}
__device__ __forceinline__ uint8_t p10dcp_depth(uint32_t x) {
    return uint8_t((x >> P10DCP_DEPTH_SHIFT) & 15u);
}

__device__ __forceinline__ BkczPlan p10dcp_build_low_plan(
    MateID d, int fixed_he, int p, uint32_t desc
) {
    BkczPlan z{};
    if (!p10dcp_active(desc) || mpair(d, p) != NN) return z;
    MateID x = msetpair(d, p, RL);
    bkcz_plan_add_low(z, x, fixed_he);
    uint16_t lm = p10dcp_lm(desc), rm = p10dcp_rm(desc);
    while (lm) {
        int i = __ffs(int(lm)) - 1;
        lm = uint16_t(lm & (lm - 1));
        int q = p - 2 - i;
        x = msetpair(d, p, LL);
        x = mset(x, q, R);
        bkcz_plan_add_low(z, x, fixed_he);
    }
    while (rm) {
        int i = __ffs(int(rm)) - 1;
        rm = uint16_t(rm & (rm - 1));
        int q = p + 1 + i;
        x = msetpair(d, p, RR);
        x = mset(x, q, L);
        bkcz_plan_add_low(z, x, fixed_he);
    }
    uint32_t depth = p10dcp_depth(desc);
    if (depth) {
        x = msetpair(d, p, RR);
        uint32_t loc = 0, bid = 0;
        if (bkcz_low_source_ref(x, fixed_he + 2, loc, bid))
            bkcz_plan_set_cross(z, bid, loc, depth);
    }
    return z;
}

__device__ __forceinline__ BkczPlan p10dcp_build_high_plan(
    MateID d, int fixed_hs, int rel, uint32_t desc
) {
    BkczPlan z{};
    if (!p10dcp_active(desc) || mpair(d, rel) != NN) return z;
    MateID x = msetpair(d, rel, RL);
    bkcz_plan_add_high(z, x, fixed_hs);
    uint16_t lm = p10dcp_lm(desc), rm = p10dcp_rm(desc);
    while (lm) {
        int i = __ffs(int(lm)) - 1;
        lm = uint16_t(lm & (lm - 1));
        int q = rel - 2 - i;
        x = msetpair(d, rel, LL);
        x = mset(x, q, R);
        bkcz_plan_add_high(z, x, fixed_hs);
    }
    while (rm) {
        int i = __ffs(int(rm)) - 1;
        rm = uint16_t(rm & (rm - 1));
        int q = rel + 1 + i;
        x = msetpair(d, rel, RR);
        x = mset(x, q, L);
        bkcz_plan_add_high(z, x, fixed_hs);
    }
    uint32_t depth = p10dcp_depth(desc);
    if (depth) {
        x = msetpair(d, rel, LL);
        uint32_t loc = 0, bid = 0;
        if (bkcz_high_source_ref(x, fixed_hs + 2, loc, bid))
            bkcz_plan_set_cross(z, bid, loc, depth);
    }
    return z;
}

__device__ __forceinline__ BkczPlan p10dcp_forward_low(
    uint32_t desc, uint32_t loc, const BucketPhysicalBlock& db, int p
) {
    uint32_t dc = bkci_low_code(loc, db.hs);
    MateID d = p == 1
        ? (MateID(dc) | (MateID(db.c) << (2 * LOW_LUT_K)))
        : minsert(MateID(dc), p, N);
    return p10dcp_build_low_plan(d, db.he, p, desc);
}
__device__ __forceinline__ BkczPlan p10dcp_forward_high(
    uint32_t desc, uint32_t loc, const BucketPhysicalBlock& db, int p
) {
    uint32_t dc = bkci_high_code(loc, db.he);
    int rel = p - LOW_LUT_K;
    MateID d = minsert(MateID(dc), rel, N);
    return p10dcp_build_high_plan(d, db.hs, rel, desc);
}
__device__ __forceinline__ BkczPlan p10dcp_reverse_low(
    uint32_t desc, uint32_t loc, const BucketPhysicalBlock& db, int p
) {
    uint32_t dc = bkci_low_code(loc, db.hs);
    MateID d = blocked_exclude_reverse(MateID(dc), LOW_LUT_K + 1, p);
    return p10dcp_build_low_plan(d, db.he, p, desc);
}
__device__ __forceinline__ BkczPlan p10dcp_reverse_high(
    uint32_t desc, uint32_t loc, const BucketPhysicalBlock& db, int p, bool edge
) {
    uint32_t dc = bkci_high_code(loc, db.he);
    int rel = p - LOW_LUT_K;
    MateID d = edge
        ? (MateID(db.c) | (MateID(dc) << 2))
        : blocked_exclude_reverse(MateID(dc), HIGH_LUT_K + 1, rel);
    return p10dcp_build_high_plan(d, db.hs, rel, desc);
}

__global__ void bucket_low_orbit_closure_pattern10_depthcode_predecoded_kernel(int p) {
    uint32_t bid = blockIdx.z;
    if (bid >= D_BKF_MAIN_NBLOCKS) return;
    uint32_t pi = uint32_t(LOW_LUT_K - p);
    size_t oi = size_t(pi) * D_BKF_LOW_PITCH + bid;
    uint32_t na = D_BKF_LOW_NN_OFF[oi], nb = D_BKF_LOW_NN_OFF[oi + 1];
    uint32_t ra = D_BKF_LOW_NR_OFF[oi], rb = D_BKF_LOW_NR_OFF[oi + 1];
    uint32_t la = D_BKF_LOW_NL_OFF[oi], lb = D_BKF_LOW_NL_OFF[oi + 1];
    uint32_t n0 = nb - na, n1 = rb - ra, total = n0 + n1 + (lb - la);
    if (!total) return;
    for (uint32_t k = uint32_t(blockIdx.x) * blockDim.x + threadIdx.x;
         k < total; k += uint32_t(gridDim.x) * blockDim.x) {
        uint32_t kind, qi, sid;
        BucketOrbitOp op;
        if (k < n0) { kind = CPU_ORBIT_NN; sid = 0; qi = na + k; op = D_BKF_LOW_NN[qi]; }
        else if (k < n0 + n1) { kind = CPU_ORBIT_NR; sid = 1; qi = ra + k - n0; op = D_BKF_LOW_NR[qi]; }
        else { kind = CPU_ORBIT_NL; sid = 2; qi = la + k - n0 - n1; op = D_BKF_LOW_NL[qi]; }
        uint32_t sl = bkf_orbit_src(op), jl = bkf_orbit_partner(op), dl = bkf_orbit_drop(op);
        uint32_t ss = bkf_loc_owner(sl), js = bkf_loc_owner(jl), ds = bkf_loc_owner(dl);
        BucketPhysicalBlock xb = bkf_low_main(ss, bid);
        if (!xb.valid || !xb.rows || !xb.cols) continue;
        uint32_t desc = p10dcp_desc(op, false, false, sid, p, uint32_t(xb.he));
        uint32_t jbid = bid;
        if (p == LOW_LUT_K) {
            uint32_t center = kind == CPU_ORBIT_NR ? uint32_t(R) : uint32_t(::L);
            jbid = 3u * uint32_t(xb.he) + center;
        }
        BucketPhysicalBlock jb = bkf_low_main(js, jbid), db = bkf_low_block(ds, uint32_t(xb.he));
        uint32_t sr = bkf_loc_rank(sl), jr = bkf_loc_rank(jl), dr = bkf_loc_rank(dl);
        BkczPlan plan = p == 1 ? p10dcp_forward_low(desc, sl, xb, p)
                               : p10dcp_forward_low(desc, dl, db, p);
        for (uint32_t hr = blockIdx.y; hr < xb.rows; hr += gridDim.y) {
            Count* ip = bkf_ptr(ss, xb.off + Code(hr) * xb.cols + sr);
            Count* jp = bkf_ptr(js, jb.off + Code(hr) * jb.cols + jr);
            Count* dp = bkf_ptr(ds, db.off + Code(hr) * db.cols + dr);
            Count c = *ip, old = *dp, extra = bkcz_low_plan_sum(plan, p == 1 ? xb : db, hr);
            if (kind == CPU_ORBIT_NN) {
                *jp = gpu_direct_add(*jp, c);
                *ip = gpu_direct_add(gpu_direct_add(c, old), p == 1 ? extra : 0);
                *dp = p == 1 ? 0 : extra;
            } else {
                Count cc = *jp, all = gpu_direct_add(gpu_direct_add(c, cc), old);
                if (p == 1) { *ip = all; *jp = gpu_direct_add(c, cc); *dp = 0; }
                else { *ip = all; *dp = gpu_direct_add(c, extra); }
            }
        }
    }
}

__global__ void bucket_high_orbit_closure_pattern10_depthcode_predecoded_kernel(int p) {
    uint32_t bid = blockIdx.z;
    if (bid >= D_BKF_MAIN_NBLOCKS) return;
    uint32_t pi = uint32_t((TARGET_W - 1) - p);
    size_t oi = size_t(pi) * D_BKF_HIGH_PITCH + bid;
    uint32_t na = D_BKF_HIGH_NN_OFF[oi], nb = D_BKF_HIGH_NN_OFF[oi + 1];
    uint32_t ra = D_BKF_HIGH_NRNL_OFF[oi], rb = D_BKF_HIGH_NRNL_OFF[oi + 1];
    uint32_t n0 = nb - na, total = n0 + (rb - ra);
    if (!total) return;
    __shared__ BkczPlan plan;
    for (uint32_t k = blockIdx.y; k < total; k += gridDim.y) {
        bool nn = k < n0;
        uint32_t qi = nn ? na + k : ra + k - n0, sid = nn ? 0u : 3u;
        BucketOrbitOp op = nn ? D_BKF_HIGH_NN[qi] : D_BKF_HIGH_NRNL[qi];
        uint32_t sl = bkf_orbit_src(op), jl = bkf_orbit_partner(op), dl = bkf_orbit_drop(op);
        uint32_t ss = bkf_loc_owner(sl), js = bkf_loc_owner(jl), ds = bkf_loc_owner(dl);
        BucketPhysicalBlock xb = bkf_high_main(ss, bid);
        if (!xb.valid || !xb.rows || !xb.cols) continue;
        uint32_t jbid = bid;
        if (p == LOW_LUT_K + 1) {
            uint32_t center = nn ? uint32_t(R) : uint32_t(N);
            int he = int(xb.hs) + (center == uint32_t(R) ? 1 : 0);
            jbid = uint32_t(3 * he + int(center));
        }
        BucketPhysicalBlock jb = bkf_high_main(js, jbid), db = bkf_high_block(ds, uint32_t(xb.hs));
        uint32_t sr = bkf_loc_rank(sl), jr = bkf_loc_rank(jl), dr = bkf_loc_rank(dl);
        if (threadIdx.x == 0) {
            uint32_t desc = p10dcp_desc(op, false, true, sid, p, uint32_t(xb.hs));
            plan = p10dcp_forward_high(desc, dl, db, p);
        }
        __syncthreads();
        for (uint32_t lr = uint32_t(blockIdx.x) * blockDim.x + threadIdx.x;
             lr < xb.cols; lr += uint32_t(gridDim.x) * blockDim.x) {
            Count* ip = bkf_ptr(ss, xb.off + Code(sr) * xb.cols + lr);
            Count* jp = bkf_ptr(js, jb.off + Code(jr) * jb.cols + lr);
            Count* dp = bkf_ptr(ds, db.off + Code(dr) * db.cols + lr);
            Count c = *ip, old = *dp, extra = bkcz_high_plan_sum(plan, db, lr);
            if (nn) { *jp = gpu_direct_add(*jp, c); *ip = gpu_direct_add(c, old); *dp = extra; }
            else { Count cc = *jp; *ip = gpu_direct_add(gpu_direct_add(c, cc), old); *dp = gpu_direct_add(c, extra); }
        }
        __syncthreads();
    }
}

__global__ void bucket_reverse_low_pattern10_depthcode_predecoded_kernel(int p) {
    uint32_t bid = blockIdx.z;
    if (bid >= D_BKF_MAIN_NBLOCKS) return;
    uint32_t pi = uint32_t(p - 1);
    size_t oi = size_t(pi) * D_RS54_PITCH + bid;
    uint32_t na = D_RS54_LOW_NN_OFF[oi], nb = D_RS54_LOW_NN_OFF[oi + 1];
    uint32_t ra = D_RS54_LOW_NR_OFF[oi], rb = D_RS54_LOW_NR_OFF[oi + 1];
    uint32_t la = D_RS54_LOW_NL_OFF[oi], lb = D_RS54_LOW_NL_OFF[oi + 1];
    uint32_t n0 = nb - na, n1 = rb - ra, total = n0 + n1 + (lb - la);
    if (!total) return;
    for (uint32_t k = uint32_t(blockIdx.x) * blockDim.x + threadIdx.x;
         k < total; k += uint32_t(gridDim.x) * blockDim.x) {
        uint32_t kind, qi, sid;
        BucketOrbitOp op;
        if (k < n0) { kind = CPU_ORBIT_NN; sid = 0; qi = na + k; op = D_RS54_LOW_NN[qi]; }
        else if (k < n0 + n1) { kind = CPU_ORBIT_NR; sid = 1; qi = ra + k - n0; op = D_RS54_LOW_NR[qi]; }
        else { kind = CPU_ORBIT_NL; sid = 2; qi = la + k - n0 - n1; op = D_RS54_LOW_NL[qi]; }
        uint32_t sl = bkf_orbit_src(op), jl = bkf_orbit_partner(op), dl = bkf_orbit_drop(op);
        uint32_t ss = bkf_loc_owner(sl), js = bkf_loc_owner(jl), ds = bkf_loc_owner(dl);
        BucketPhysicalBlock xb = bkf_low_main(ss, bid);
        BucketPhysicalBlock jb = bkf_low_main(js, bkcp10_reverse_low_jblock(bid, xb, p, kind));
        BucketPhysicalBlock db = bkf_low_block(ds, uint32_t(xb.he));
        uint32_t desc = p10dcp_desc(op, true, false, sid, p, uint32_t(xb.he));
        BkczPlan plan = p10dcp_reverse_low(desc, dl, db, p);
        for (uint32_t hr = blockIdx.y; hr < xb.rows; hr += gridDim.y) {
            Count* ip = bkf_ptr(ss, xb.off + Code(hr) * xb.cols + bkf_loc_rank(sl));
            Count* jp = bkf_ptr(js, jb.off + Code(hr) * jb.cols + bkf_loc_rank(jl));
            Count* dp = bkf_ptr(ds, db.off + Code(hr) * db.cols + bkf_loc_rank(dl));
            Count c = *ip, old = *dp, extra = bkcz_low_plan_sum(plan, db, hr);
            if (kind == CPU_ORBIT_NN) { *jp = gpu_direct_add(*jp, c); *ip = gpu_direct_add(c, old); *dp = extra; }
            else { Count cc = *jp; *ip = gpu_direct_add(gpu_direct_add(c, cc), old); *dp = gpu_direct_add(c, extra); }
        }
    }
}

__global__ void bucket_reverse_high_pattern10_depthcode_predecoded_kernel(int p) {
    uint32_t bid = blockIdx.z;
    if (bid >= D_BKF_MAIN_NBLOCKS) return;
    uint32_t pi = uint32_t(p - (LOW_LUT_K + 1));
    size_t oi = size_t(pi) * D_RS54_PITCH + bid;
    uint32_t na = D_RS54_HIGH_NN_OFF[oi], nb = D_RS54_HIGH_NN_OFF[oi + 1];
    uint32_t ra = D_RS54_HIGH_NR_OFF[oi], rb = D_RS54_HIGH_NR_OFF[oi + 1];
    uint32_t la = D_RS54_HIGH_NL_OFF[oi], lb = D_RS54_HIGH_NL_OFF[oi + 1];
    uint32_t n0 = nb - na, n1 = rb - ra, total = n0 + n1 + (lb - la);
    if (!total) return;
    bool edge = p == TARGET_W - 1;
    __shared__ BkczPlan plan;
    for (uint32_t k = blockIdx.y; k < total; k += gridDim.y) {
        uint32_t kind, qi, sid;
        BucketOrbitOp op;
        if (k < n0) { kind = CPU_ORBIT_NN; sid = 0; qi = na + k; op = D_RS54_HIGH_NN[qi]; }
        else if (k < n0 + n1) { kind = CPU_ORBIT_NR; sid = 1; qi = ra + k - n0; op = D_RS54_HIGH_NR[qi]; }
        else { kind = CPU_ORBIT_NL; sid = 2; qi = la + k - n0 - n1; op = D_RS54_HIGH_NL[qi]; }
        uint32_t sl = bkf_orbit_src(op), jl = bkf_orbit_partner(op), dl = bkf_orbit_drop(op);
        uint32_t ss = bkf_loc_owner(sl), js = bkf_loc_owner(jl), ds = bkf_loc_owner(dl);
        BucketPhysicalBlock xb = bkf_high_main(ss, bid);
        BucketPhysicalBlock jb = bkf_high_main(js, bkcp10_reverse_high_jblock(bid, xb, p, kind));
        BucketPhysicalBlock db = bkf_high_block(ds, uint32_t(xb.hs));
        if (threadIdx.x == 0) {
            uint32_t desc = p10dcp_desc(op, true, true, sid, p, uint32_t(xb.hs));
            plan = edge ? p10dcp_reverse_high(desc, sl, xb, p, true)
                        : p10dcp_reverse_high(desc, dl, db, p, false);
        }
        __syncthreads();
        for (uint32_t lr = uint32_t(blockIdx.x) * blockDim.x + threadIdx.x;
             lr < xb.cols; lr += uint32_t(gridDim.x) * blockDim.x) {
            Count* ip = bkf_ptr(ss, xb.off + Code(bkf_loc_rank(sl)) * xb.cols + lr);
            Count* jp = bkf_ptr(js, jb.off + Code(bkf_loc_rank(jl)) * jb.cols + lr);
            Count* dp = bkf_ptr(ds, db.off + Code(bkf_loc_rank(dl)) * db.cols + lr);
            Count c = *ip, old = *dp, extra = bkcz_high_plan_sum(plan, edge ? xb : db, lr);
            if (kind == CPU_ORBIT_NN) {
                *jp = gpu_direct_add(*jp, c);
                *ip = gpu_direct_add(gpu_direct_add(c, old), edge ? extra : 0);
                *dp = edge ? 0 : extra;
            } else {
                Count cc = *jp;
                *ip = gpu_direct_add(gpu_direct_add(c, cc), old);
                if (edge) { *jp = gpu_direct_add(c, cc); *dp = 0; }
                else *dp = gpu_direct_add(c, extra);
            }
        }
        __syncthreads();
    }
}

static void bucket_launch_low_orbit_closure_pattern10_depthcode_predecoded(
    const StorageLayout& layout, int threads = 256, int gx = 16, int gy = 8
) {
    dim3 block(threads), grid(gx, gy, unsigned(layout.main_blocks.size()));
    for (int p = LOW_LUT_K; p >= 1; --p) {
        bucket_low_orbit_closure_pattern10_depthcode_predecoded_kernel<<<grid, block>>>(p);
        ck(cudaGetLastError(), "bucket low pattern10 depthcode predecoded");
    }
}
static void bucket_launch_high_orbit_closure_pattern10_depthcode_predecoded(
    const StorageLayout& layout, int threads = 256, int gx = 16, int gy = 8
) {
    dim3 block(threads), grid(gx, gy, unsigned(layout.main_blocks.size()));
    for (int p = TARGET_W - 1; p >= LOW_LUT_K + 1; --p) {
        bucket_high_orbit_closure_pattern10_depthcode_predecoded_kernel<<<grid, block>>>(p);
        ck(cudaGetLastError(), "bucket high pattern10 depthcode predecoded");
    }
}
static void bucket_launch_reverse_low_pattern10_depthcode_predecoded(
    const StorageLayout& layout, int threads = 256, int gx = 16, int gy = 8
) {
    dim3 block(threads), grid(gx, gy, unsigned(layout.main_blocks.size()));
    for (int p = 1; p <= LOW_LUT_K; ++p) {
        bucket_reverse_low_pattern10_depthcode_predecoded_kernel<<<grid, block>>>(p);
        ck(cudaGetLastError(), "bucket reverse low pattern10 depthcode predecoded");
    }
}
static void bucket_launch_reverse_high_pattern10_depthcode_predecoded(
    const StorageLayout& layout, int threads = 256, int gx = 16, int gy = 8
) {
    dim3 block(threads), grid(gx, gy, unsigned(layout.main_blocks.size()));
    for (int p = LOW_LUT_K + 1; p < TARGET_W; ++p) {
        bucket_reverse_high_pattern10_depthcode_predecoded_kernel<<<grid, block>>>(p);
        ck(cudaGetLastError(), "bucket reverse high pattern10 depthcode predecoded");
    }
}
