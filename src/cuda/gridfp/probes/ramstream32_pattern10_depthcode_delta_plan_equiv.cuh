#pragma once

#include "../ramstream32_bucket_closure_pattern10_depthcode_delta_plan.cuh"

__device__ __forceinline__ bool p10dc_delta_plan_equal(const BkczPlan& a, const BkczPlan& b) {
    if (a.meta != b.meta) return false;
#pragma unroll
    for (uint32_t i = 0; i < BKCZ_MAX_LOCAL; ++i)
        if (a.local[i] != b.local[i]) return false;
    return true;
}

__device__ __forceinline__ void p10dc_delta_account(
    const BkczPlan& canonical, const BkczPlan& delta,
    unsigned long long* checked, unsigned long long* mismatches
) {
    atomicAdd(checked, 1ull);
    if (!p10dc_delta_plan_equal(canonical, delta)) atomicAdd(mismatches, 1ull);
}

__global__ void p10dc_delta_equiv_forward_low_kernel(
    int p, unsigned long long* checked, unsigned long long* mismatches
) {
    uint32_t bid = blockIdx.x;
    if (bid >= D_BKF_MAIN_NBLOCKS) return;
    uint32_t pi = uint32_t(LOW_LUT_K - p);
    size_t oi = size_t(pi) * D_BKF_LOW_PITCH + bid;
    uint32_t na = D_BKF_LOW_NN_OFF[oi], nb = D_BKF_LOW_NN_OFF[oi + 1];
    uint32_t ra = D_BKF_LOW_NR_OFF[oi], rb = D_BKF_LOW_NR_OFF[oi + 1];
    uint32_t la = D_BKF_LOW_NL_OFF[oi], lb = D_BKF_LOW_NL_OFF[oi + 1];
    uint32_t n0 = nb - na, n1 = rb - ra, total = n0 + n1 + (lb - la);
    for (uint32_t k = threadIdx.x; k < total; k += blockDim.x) {
        uint32_t sid = 0, qi = 0;
        BucketOrbitOp op;
        if (k < n0) { sid = 0; qi = na + k; op = D_BKF_LOW_NN[qi]; }
        else if (k < n0 + n1) { sid = 1; qi = ra + k - n0; op = D_BKF_LOW_NR[qi]; }
        else { sid = 2; qi = la + k - n0 - n1; op = D_BKF_LOW_NL[qi]; }
        uint32_t sl = bkf_orbit_src(op), dl = bkf_orbit_drop(op);
        uint32_t ss = bkf_loc_owner(sl), ds = bkf_loc_owner(dl);
        BucketPhysicalBlock xb = bkf_low_main(ss, bid);
        if (!xb.valid || !xb.rows || !xb.cols) continue;
        BucketPhysicalBlock db = bkf_low_block(ds, uint32_t(xb.he));
        uint32_t payload = p10dc_payload(op, false, false, sid, p, uint32_t(xb.he));
        BkczPlan a = p == 1 ? p10dc_forward_low(payload, sl, xb, p)
                            : p10dc_forward_low(payload, dl, db, p);
        BkczPlan b = p == 1 ? p10dc_forward_low_delta(payload, sl, xb, p)
                            : p10dc_forward_low_delta(payload, dl, db, p);
        p10dc_delta_account(a, b, checked, mismatches);
    }
}

__global__ void p10dc_delta_equiv_forward_high_kernel(
    int p, unsigned long long* checked, unsigned long long* mismatches
) {
    uint32_t bid = blockIdx.x;
    if (bid >= D_BKF_MAIN_NBLOCKS) return;
    uint32_t pi = uint32_t((TARGET_W - 1) - p);
    size_t oi = size_t(pi) * D_BKF_HIGH_PITCH + bid;
    uint32_t na = D_BKF_HIGH_NN_OFF[oi], nb = D_BKF_HIGH_NN_OFF[oi + 1];
    uint32_t ra = D_BKF_HIGH_NRNL_OFF[oi], rb = D_BKF_HIGH_NRNL_OFF[oi + 1];
    uint32_t n0 = nb - na, total = n0 + (rb - ra);
    for (uint32_t k = threadIdx.x; k < total; k += blockDim.x) {
        bool nn = k < n0;
        uint32_t sid = nn ? 0u : 3u;
        uint32_t qi = nn ? na + k : ra + k - n0;
        BucketOrbitOp op = nn ? D_BKF_HIGH_NN[qi] : D_BKF_HIGH_NRNL[qi];
        uint32_t sl = bkf_orbit_src(op), dl = bkf_orbit_drop(op);
        uint32_t ss = bkf_loc_owner(sl), ds = bkf_loc_owner(dl);
        BucketPhysicalBlock xb = bkf_high_main(ss, bid);
        if (!xb.valid || !xb.rows || !xb.cols) continue;
        BucketPhysicalBlock db = bkf_high_block(ds, uint32_t(xb.hs));
        uint32_t payload = p10dc_payload(op, false, true, sid, p, uint32_t(xb.hs));
        BkczPlan a = p10dc_forward_high(payload, dl, db, p);
        BkczPlan b = p10dc_forward_high_delta(payload, dl, db, p);
        p10dc_delta_account(a, b, checked, mismatches);
    }
}

__global__ void p10dc_delta_equiv_reverse_low_kernel(
    int p, unsigned long long* checked, unsigned long long* mismatches
) {
    uint32_t bid = blockIdx.x;
    if (bid >= D_BKF_MAIN_NBLOCKS) return;
    uint32_t pi = uint32_t(p - 1);
    size_t oi = size_t(pi) * D_RS54_PITCH + bid;
    uint32_t na = D_RS54_LOW_NN_OFF[oi], nb = D_RS54_LOW_NN_OFF[oi + 1];
    uint32_t ra = D_RS54_LOW_NR_OFF[oi], rb = D_RS54_LOW_NR_OFF[oi + 1];
    uint32_t la = D_RS54_LOW_NL_OFF[oi], lb = D_RS54_LOW_NL_OFF[oi + 1];
    uint32_t n0 = nb - na, n1 = rb - ra, total = n0 + n1 + (lb - la);
    for (uint32_t k = threadIdx.x; k < total; k += blockDim.x) {
        uint32_t sid = 0, qi = 0;
        BucketOrbitOp op;
        if (k < n0) { sid = 0; qi = na + k; op = D_RS54_LOW_NN[qi]; }
        else if (k < n0 + n1) { sid = 1; qi = ra + k - n0; op = D_RS54_LOW_NR[qi]; }
        else { sid = 2; qi = la + k - n0 - n1; op = D_RS54_LOW_NL[qi]; }
        uint32_t sl = bkf_orbit_src(op), dl = bkf_orbit_drop(op);
        uint32_t ss = bkf_loc_owner(sl), ds = bkf_loc_owner(dl);
        BucketPhysicalBlock xb = bkf_low_main(ss, bid);
        if (!xb.valid || !xb.rows || !xb.cols) continue;
        BucketPhysicalBlock db = bkf_low_block(ds, uint32_t(xb.he));
        uint32_t payload = p10dc_payload(op, true, false, sid, p, uint32_t(xb.he));
        BkczPlan a = p10dc_reverse_low(payload, dl, db, p);
        BkczPlan b = p10dc_reverse_low_delta(payload, dl, db, p);
        p10dc_delta_account(a, b, checked, mismatches);
    }
}

__global__ void p10dc_delta_equiv_reverse_high_kernel(
    int p, unsigned long long* checked, unsigned long long* mismatches
) {
    uint32_t bid = blockIdx.x;
    if (bid >= D_BKF_MAIN_NBLOCKS) return;
    uint32_t pi = uint32_t(p - (LOW_LUT_K + 1));
    size_t oi = size_t(pi) * D_RS54_PITCH + bid;
    bool edge = p == TARGET_W - 1;
    uint32_t na = D_RS54_HIGH_NN_OFF[oi], nb = D_RS54_HIGH_NN_OFF[oi + 1];
    uint32_t ra = D_RS54_HIGH_NR_OFF[oi], rb = D_RS54_HIGH_NR_OFF[oi + 1];
    uint32_t la = D_RS54_HIGH_NL_OFF[oi], lb = D_RS54_HIGH_NL_OFF[oi + 1];
    uint32_t n0 = nb - na, n1 = rb - ra, total = n0 + n1 + (lb - la);
    for (uint32_t k = threadIdx.x; k < total; k += blockDim.x) {
        uint32_t sid = 0, kind = 0, qi = 0;
        BucketOrbitOp op;
        if (k < n0) { kind = CPU_ORBIT_NN; sid = 0; qi = na + k; op = D_RS54_HIGH_NN[qi]; }
        else if (k < n0 + n1) { kind = CPU_ORBIT_NR; sid = 1; qi = ra + k - n0; op = D_RS54_HIGH_NR[qi]; }
        else { kind = CPU_ORBIT_NL; sid = 2; qi = la + k - n0 - n1; op = D_RS54_HIGH_NL[qi]; }
        (void)kind;
        uint32_t sl = bkf_orbit_src(op), dl = bkf_orbit_drop(op);
        uint32_t ss = bkf_loc_owner(sl), ds = bkf_loc_owner(dl);
        BucketPhysicalBlock xb = bkf_high_main(ss, bid);
        if (!xb.valid || !xb.rows || !xb.cols) continue;
        BucketPhysicalBlock db = bkf_high_block(ds, uint32_t(xb.hs));
        uint32_t payload = p10dc_payload(op, true, true, sid, p, uint32_t(xb.hs));
        BkczPlan a = edge ? p10dc_reverse_high(payload, sl, xb, p, true)
                          : p10dc_reverse_high(payload, dl, db, p, false);
        BkczPlan b = edge ? p10dc_reverse_high_delta(payload, sl, xb, p, true)
                          : p10dc_reverse_high_delta(payload, dl, db, p, false);
        p10dc_delta_account(a, b, checked, mismatches);
    }
}

static bool p10dc_run_delta_plan_equiv(
    const StorageLayout& layout, const BucketPhysicalLayoutHost& phy,
    BucketFusedDeviceTables& dt
) {
    unsigned long long *d_checked = nullptr, *d_mismatch = nullptr;
    ck(cudaMalloc(&d_checked, sizeof(unsigned long long)), "p10dc delta checked alloc");
    ck(cudaMalloc(&d_mismatch, sizeof(unsigned long long)), "p10dc delta mismatch alloc");
    ck(cudaMemset(d_checked, 0, sizeof(unsigned long long)), "p10dc delta checked zero");
    ck(cudaMemset(d_mismatch, 0, sizeof(unsigned long long)), "p10dc delta mismatch zero");
    dim3 block(128), grid(unsigned(layout.main_blocks.size()));

    for (uint32_t fixed = 0; fixed < BUCKET_NGPU; ++fixed) {
        std::array<Count*, BUCKET_NGPU> slots{};
        bkft_alloc_slots(fixed, phy, slots);
        dt.bind_owner(fixed, phy, slots);
        for (int p = LOW_LUT_K; p >= 1; --p)
            p10dc_delta_equiv_forward_low_kernel<<<grid, block>>>(p, d_checked, d_mismatch);
        for (int p = TARGET_W - 1; p >= LOW_LUT_K + 1; --p)
            p10dc_delta_equiv_forward_high_kernel<<<grid, block>>>(p, d_checked, d_mismatch);
        for (int p = 1; p <= LOW_LUT_K; ++p)
            p10dc_delta_equiv_reverse_low_kernel<<<grid, block>>>(p, d_checked, d_mismatch);
        for (int p = LOW_LUT_K + 1; p < TARGET_W; ++p)
            p10dc_delta_equiv_reverse_high_kernel<<<grid, block>>>(p, d_checked, d_mismatch);
        ck(cudaGetLastError(), "p10dc delta plan equivalence launch");
        ck(cudaDeviceSynchronize(), "p10dc delta plan equivalence sync");
        bkft_free_slots(slots);
    }

    unsigned long long checked = 0, mismatch = 0;
    ck(cudaMemcpy(&checked, d_checked, sizeof(checked), cudaMemcpyDeviceToHost), "p10dc delta checked D2H");
    ck(cudaMemcpy(&mismatch, d_mismatch, sizeof(mismatch), cudaMemcpyDeviceToHost), "p10dc delta mismatch D2H");
    cudaFree(d_mismatch);
    cudaFree(d_checked);
    std::cout << "pattern10-depthcode-delta-plan-equivalence checked=" << checked
              << " mismatches=" << mismatch
              << " phases=forward-low,forward-high,reverse-low,reverse-high"
              << " owners=" << BUCKET_NGPU
              << " plan_exact=" << (mismatch == 0 && checked ? 1 : 0) << '\n';
    return checked != 0 && mismatch == 0;
}
