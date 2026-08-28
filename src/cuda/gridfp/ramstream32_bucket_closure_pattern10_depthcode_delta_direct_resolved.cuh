#pragma once

#include "ramstream32_bucket_orbit_closure_pattern10_depthcode_resolved.cuh"
#include "ramstream32_bucket_closure_cross5.cuh"

// Experimental HIGH-only direct resolver. The canonical delta builder first
// packs every closure source into BkczPlan.local[], then p10dc_resolve_high_rows
// immediately unpacks those descriptors to produce row pointers. This helper
// resolves the direct-table result straight into local_base[]/cross_base and
// keeps only CROSS depth in plan.meta. Orbit format and depthcode payload stay
// unchanged.

__device__ __forceinline__ void p10dc_direct_resolve_high_io(
    P10DCHighResolvedCtx& c,
    uint32_t ss, uint32_t js, uint32_t ds,
    uint32_t sr, uint32_t jr, uint32_t dr
) {
    c.ip_base = bkf_ptr(ss, c.xb.off + Code(sr) * c.xb.cols);
    c.jp_base = bkf_ptr(js, c.jb.off + Code(jr) * c.jb.cols);
    c.dp_base = bkf_ptr(ds, c.db.off + Code(dr) * c.db.cols);
}

__device__ __forceinline__ bool p10dc_direct_add_high_base(
    P10DCHighResolvedCtx& c, uint32_t key, uint32_t center, int fixed_hs
) {
    if (fixed_hs < 0 || fixed_hs > LOW_LUT_K + 1 || center > uint32_t(::L)) return false;
    int he = fixed_hs - bkci_delta(center);
    if (he < 0 || he > HIGH_LUT_K + 1) return false;
    uint32_t z = D_BKF_HIGH_DIRECT[key];
    if (z == BKF_DIRECT_INVALID || int(bkf_direct_height(z)) != he) return false;
    uint32_t loc = bkf_direct_locator(z), owner = bkf_loc_owner(loc);
    uint32_t bid = uint32_t(3 * he + int(center));
    if (bid >= D_BKF_MAIN_NBLOCKS) return false;
    BucketPhysicalBlock sb = bkf_high_main(owner, bid);
    if (!sb.valid) return false;
    uint32_t n = uint32_t(c.local_n);
    if (n >= BKCZ_MAX_LOCAL) return false;
    c.local_base[n] = bkf_ptr(owner, sb.off + Code(bkf_loc_rank(loc)) * sb.cols);
    c.local_n = uint8_t(n + 1u);
    return true;
}

__device__ __forceinline__ void p10dc_direct_set_high_cross_base(
    P10DCHighResolvedCtx& c, uint32_t key, uint32_t center, int fixed_hs, uint32_t depth
) {
    if (!depth || fixed_hs < 0 || fixed_hs > LOW_LUT_K + 1 || center > uint32_t(::L)) return;
    int he = fixed_hs - bkci_delta(center);
    if (he < 0 || he > HIGH_LUT_K + 1) return;
    uint32_t z = D_BKF_HIGH_DIRECT[key];
    if (z == BKF_DIRECT_INVALID || int(bkf_direct_height(z)) != he) return;
    uint32_t loc = bkf_direct_locator(z), owner = bkf_loc_owner(loc);
    uint32_t bid = uint32_t(3 * he + int(center));
    if (bid >= D_BKF_MAIN_NBLOCKS) return;
    BucketPhysicalBlock sb = bkf_high_main(owner, bid);
    if (!sb.valid) return;
    c.cross_base = bkf_ptr(owner, sb.off + Code(bkf_loc_rank(loc)) * sb.cols);
    c.plan.meta = uint32_t(depth) << BKCZ_META_DEPTH_SHIFT;
}

__device__ __forceinline__ void p10dc_build_high_resolved_delta_direct(
    P10DCHighResolvedCtx& c, MateID d, int fixed_hs, int p, uint32_t payload
) {
    c.plan = BkczPlan{};
    c.local_n = 0;
    c.cross_base = nullptr;
    if (!p10dc_payload_valid(payload) || mpair(d, p) != NN) return;
    constexpr uint64_t MASK = (uint64_t(1) << (2 * HIGH_LUT_K)) - 1ull;
    uint32_t factor = uint32_t((d >> 2) & MASK);
    uint32_t base = bkcz_ternary_key<HIGH_LUT_K>(factor);
    uint32_t dest_center = uint32_t(mget(d, 0));

    {
        int32_t delta = p10dc_high_pair_delta(RL, p);
        uint32_t center = p10dc_high_center_after_pair(dest_center, RL, p);
        p10dc_direct_add_high_base(c, uint32_t(int32_t(base) + delta), center, fixed_hs);
    }

    uint16_t lm = p10dc_payload_lm(payload), rm = p10dc_payload_rm(payload);
    while (lm) {
        int i = __ffs(int(lm)) - 1;
        lm = uint16_t(lm & (lm - 1));
        int q = p - 2 - i;
        int32_t delta = p10dc_high_pair_delta(LL, p);
        uint32_t center = p10dc_high_center_after_pair(dest_center, LL, p);
        if (q > 0) delta -= int32_t(p10dc_pow3(uint32_t(q - 1)));
        else center = uint32_t(R);
        p10dc_direct_add_high_base(c, uint32_t(int32_t(base) + delta), center, fixed_hs);
    }
    while (rm) {
        int i = __ffs(int(rm)) - 1;
        rm = uint16_t(rm & (rm - 1));
        int q = p + 1 + i;
        int32_t delta = p10dc_high_pair_delta(RR, p) + int32_t(p10dc_pow3(uint32_t(q - 1)));
        uint32_t center = p10dc_high_center_after_pair(dest_center, RR, p);
        p10dc_direct_add_high_base(c, uint32_t(int32_t(base) + delta), center, fixed_hs);
    }

    uint32_t depth = uint32_t(p10dc_payload_depth(payload));
    if (depth) {
        int32_t delta = p10dc_high_pair_delta(LL, p);
        uint32_t center = p10dc_high_center_after_pair(dest_center, LL, p);
        p10dc_direct_set_high_cross_base(
            c, uint32_t(int32_t(base) + delta), center, fixed_hs + 2, depth);
    }
}

__device__ __forceinline__ void p10dc_prepare_forward_high_delta_direct(
    P10DCHighResolvedCtx& c, uint32_t payload, uint32_t loc, int p,
    uint32_t ss, uint32_t js, uint32_t ds, uint32_t sr, uint32_t jr, uint32_t dr
) {
    p10dc_direct_resolve_high_io(c, ss, js, ds, sr, jr, dr);
    uint32_t dc = bkci_high_code(loc, c.db.he);
    int rel = p - LOW_LUT_K;
    MateID d = minsert(MateID(dc), rel, N);
    p10dc_build_high_resolved_delta_direct(c, d, c.db.hs, rel, payload);
}

__device__ __forceinline__ void p10dc_prepare_reverse_high_delta_direct(
    P10DCHighResolvedCtx& c, uint32_t payload, uint32_t loc,
    const BucketPhysicalBlock& plan_db, int p, bool edge,
    uint32_t ss, uint32_t js, uint32_t ds, uint32_t sr, uint32_t jr, uint32_t dr
) {
    p10dc_direct_resolve_high_io(c, ss, js, ds, sr, jr, dr);
    uint32_t dc = bkci_high_code(loc, plan_db.he);
    int rel = p - LOW_LUT_K;
    MateID d = edge
        ? (MateID(plan_db.c) | (MateID(dc) << 2))
        : blocked_exclude_reverse(MateID(dc), HIGH_LUT_K + 1, rel);
    p10dc_build_high_resolved_delta_direct(c, d, plan_db.hs, rel, payload);
}

__device__ __forceinline__ Count p10dc_direct_resolved_high_plan_sum_cross5(
    const P10DCHighResolvedCtx& c, const BucketPhysicalBlock& db, uint32_t lr
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
        sum += p10dc_resolved_low_preimages_cross5(dc, depth, c.cross_base);
#else
        sum = gpu_direct_add(sum, p10dc_resolved_low_preimages_cross5(dc, depth, c.cross_base));
#endif
    }
#if GPU_DIRECT_PM_ACCUM
    return gpu_direct_pm_reduce_u64(sum);
#else
    return sum;
#endif
}
