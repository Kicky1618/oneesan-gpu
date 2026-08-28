#pragma once

#include "ramstream32_bucket_orbit_closure_pattern10_depthcode.cuh"

// Ternary-delta closure plan reconstruction.
//
// The canonical payload already identifies every ordinary closure source by
// its LL/RR candidate masks. Instead of rebuilding a MateID for each source and
// recomputing a K-digit ternary key in bkcz_*_source_ref(), compute the
// destination ternary key once, then reach each source by an exact +/- 3^q
// update. The pair itself contributes a fixed delta because the destination
// pair is NN. Source center symbols are derived directly from the same edits.
//
// This keeps the orbit format and depthcode payload unchanged. It is a pure
// runtime reconstruction optimization and can therefore be A/B tested against
// the canonical MateID plan builder.

__device__ __constant__ uint32_t D_P10DC_POW3[14] = {
    1u, 3u, 9u, 27u, 81u, 243u, 729u,
    2187u, 6561u, 19683u, 59049u, 177147u, 531441u, 1594323u
};

__device__ __forceinline__ uint32_t p10dc_pow3(uint32_t i) {
    return D_P10DC_POW3[i];
}

__device__ __forceinline__ uint32_t p10dc_pair_lo(MateValuePair pair) {
    return uint32_t(pair) & 3u;
}
__device__ __forceinline__ uint32_t p10dc_pair_hi(MateValuePair pair) {
    return (uint32_t(pair) >> 2) & 3u;
}

__device__ __forceinline__ bool p10dc_add_low_key(
    BkczPlan& plan, uint32_t key, uint32_t center, int fixed_he
) {
    if (fixed_he < 0 || fixed_he > HIGH_LUT_K + 1 || center > uint32_t(::L)) return false;
    int hs = fixed_he + bkci_delta(center);
    if (hs < 0 || hs > LOW_LUT_K + 1) return false;
    uint32_t z = D_BKF_LOW_DIRECT[key];
    if (z == BKF_DIRECT_INVALID || int(bkf_direct_height(z)) != hs) return false;
    uint32_t loc = bkf_direct_locator(z);
    uint32_t bid = uint32_t(3 * fixed_he + int(center));
    if (bid >= D_BKF_MAIN_NBLOCKS || !bkf_low_main(bkf_loc_owner(loc), bid).valid) return false;
    uint32_t n = bkcz_plan_local_n(plan);
    if (n >= BKCZ_MAX_LOCAL) return false;
    plan.local[n] = bkcz_src_pack_device(bid, loc);
    plan.meta = (plan.meta & ~BKCZ_META_LOCAL_N_MASK) | ((n + 1u) << BKCZ_META_LOCAL_N_SHIFT);
    return true;
}

__device__ __forceinline__ bool p10dc_add_high_key(
    BkczPlan& plan, uint32_t key, uint32_t center, int fixed_hs
) {
    if (fixed_hs < 0 || fixed_hs > LOW_LUT_K + 1 || center > uint32_t(::L)) return false;
    int he = fixed_hs - bkci_delta(center);
    if (he < 0 || he > HIGH_LUT_K + 1) return false;
    uint32_t z = D_BKF_HIGH_DIRECT[key];
    if (z == BKF_DIRECT_INVALID || int(bkf_direct_height(z)) != he) return false;
    uint32_t loc = bkf_direct_locator(z);
    uint32_t bid = uint32_t(3 * he + int(center));
    if (bid >= D_BKF_MAIN_NBLOCKS || !bkf_high_main(bkf_loc_owner(loc), bid).valid) return false;
    uint32_t n = bkcz_plan_local_n(plan);
    if (n >= BKCZ_MAX_LOCAL) return false;
    plan.local[n] = bkcz_src_pack_device(bid, loc);
    plan.meta = (plan.meta & ~BKCZ_META_LOCAL_N_MASK) | ((n + 1u) << BKCZ_META_LOCAL_N_SHIFT);
    return true;
}

__device__ __forceinline__ void p10dc_set_low_cross_key(
    BkczPlan& plan, uint32_t key, uint32_t center, int fixed_he, uint32_t depth
) {
    if (!depth || fixed_he < 0 || fixed_he > HIGH_LUT_K + 1 || center > uint32_t(::L)) return;
    int hs = fixed_he + bkci_delta(center);
    if (hs < 0 || hs > LOW_LUT_K + 1) return;
    uint32_t z = D_BKF_LOW_DIRECT[key];
    if (z == BKF_DIRECT_INVALID || int(bkf_direct_height(z)) != hs) return;
    uint32_t loc = bkf_direct_locator(z);
    uint32_t bid = uint32_t(3 * fixed_he + int(center));
    if (bid >= D_BKF_MAIN_NBLOCKS || !bkf_low_main(bkf_loc_owner(loc), bid).valid) return;
    bkcz_plan_set_cross(plan, bid, loc, depth);
}

__device__ __forceinline__ void p10dc_set_high_cross_key(
    BkczPlan& plan, uint32_t key, uint32_t center, int fixed_hs, uint32_t depth
) {
    if (!depth || fixed_hs < 0 || fixed_hs > LOW_LUT_K + 1 || center > uint32_t(::L)) return;
    int he = fixed_hs - bkci_delta(center);
    if (he < 0 || he > HIGH_LUT_K + 1) return;
    uint32_t z = D_BKF_HIGH_DIRECT[key];
    if (z == BKF_DIRECT_INVALID || int(bkf_direct_height(z)) != he) return;
    uint32_t loc = bkf_direct_locator(z);
    uint32_t bid = uint32_t(3 * he + int(center));
    if (bid >= D_BKF_MAIN_NBLOCKS || !bkf_high_main(bkf_loc_owner(loc), bid).valid) return;
    bkcz_plan_set_cross(plan, bid, loc, depth);
}

__device__ __forceinline__ int32_t p10dc_low_pair_delta(MateValuePair pair, int p) {
    int32_t d = int32_t(p10dc_pair_lo(pair) * p10dc_pow3(uint32_t(p - 1)));
    if (p < LOW_LUT_K) d += int32_t(p10dc_pair_hi(pair) * p10dc_pow3(uint32_t(p)));
    return d;
}

__device__ __forceinline__ int32_t p10dc_high_pair_delta(MateValuePair pair, int p) {
    int32_t d = int32_t(p10dc_pair_hi(pair) * p10dc_pow3(uint32_t(p - 1)));
    if (p > 1) d += int32_t(p10dc_pair_lo(pair) * p10dc_pow3(uint32_t(p - 2)));
    return d;
}

__device__ __forceinline__ uint32_t p10dc_low_center_after_pair(uint32_t center, MateValuePair pair, int p) {
    return p == LOW_LUT_K ? p10dc_pair_hi(pair) : center;
}

__device__ __forceinline__ uint32_t p10dc_high_center_after_pair(uint32_t center, MateValuePair pair, int p) {
    return p == 1 ? p10dc_pair_lo(pair) : center;
}

__device__ __forceinline__ BkczPlan p10dc_build_low_plan_delta(
    MateID d, int fixed_he, int p, uint32_t payload
) {
    BkczPlan z{};
    if (!p10dc_payload_valid(payload) || mpair(d, p) != NN) return z;
    constexpr uint64_t MASK = (uint64_t(1) << (2 * LOW_LUT_K)) - 1ull;
    uint32_t factor = uint32_t(d & MASK);
    uint32_t base = bkcz_ternary_key<LOW_LUT_K>(factor);
    uint32_t dest_center = uint32_t(mget(d, LOW_LUT_K));

    {
        int32_t delta = p10dc_low_pair_delta(RL, p);
        uint32_t center = p10dc_low_center_after_pair(dest_center, RL, p);
        p10dc_add_low_key(z, uint32_t(int32_t(base) + delta), center, fixed_he);
    }

    uint16_t lm = p10dc_payload_lm(payload), rm = p10dc_payload_rm(payload);
    while (lm) {
        int i = __ffs(int(lm)) - 1;
        lm = uint16_t(lm & (lm - 1));
        int q = p - 2 - i;
        int32_t delta = p10dc_low_pair_delta(LL, p) - int32_t(p10dc_pow3(uint32_t(q)));
        uint32_t center = p10dc_low_center_after_pair(dest_center, LL, p);
        p10dc_add_low_key(z, uint32_t(int32_t(base) + delta), center, fixed_he);
    }
    while (rm) {
        int i = __ffs(int(rm)) - 1;
        rm = uint16_t(rm & (rm - 1));
        int q = p + 1 + i;
        int32_t delta = p10dc_low_pair_delta(RR, p);
        uint32_t center = p10dc_low_center_after_pair(dest_center, RR, p);
        if (q < LOW_LUT_K) delta += int32_t(p10dc_pow3(uint32_t(q)));
        else center = uint32_t(::L);
        p10dc_add_low_key(z, uint32_t(int32_t(base) + delta), center, fixed_he);
    }

    uint32_t depth = uint32_t(p10dc_payload_depth(payload));
    if (depth) {
        int32_t delta = p10dc_low_pair_delta(RR, p);
        uint32_t center = p10dc_low_center_after_pair(dest_center, RR, p);
        p10dc_set_low_cross_key(z, uint32_t(int32_t(base) + delta), center, fixed_he + 2, depth);
    }
    return z;
}

__device__ __forceinline__ BkczPlan p10dc_build_high_plan_delta(
    MateID d, int fixed_hs, int p, uint32_t payload
) {
    BkczPlan z{};
    if (!p10dc_payload_valid(payload) || mpair(d, p) != NN) return z;
    constexpr uint64_t MASK = (uint64_t(1) << (2 * HIGH_LUT_K)) - 1ull;
    uint32_t factor = uint32_t((d >> 2) & MASK);
    uint32_t base = bkcz_ternary_key<HIGH_LUT_K>(factor);
    uint32_t dest_center = uint32_t(mget(d, 0));

    {
        int32_t delta = p10dc_high_pair_delta(RL, p);
        uint32_t center = p10dc_high_center_after_pair(dest_center, RL, p);
        p10dc_add_high_key(z, uint32_t(int32_t(base) + delta), center, fixed_hs);
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
        p10dc_add_high_key(z, uint32_t(int32_t(base) + delta), center, fixed_hs);
    }
    while (rm) {
        int i = __ffs(int(rm)) - 1;
        rm = uint16_t(rm & (rm - 1));
        int q = p + 1 + i;
        int32_t delta = p10dc_high_pair_delta(RR, p) + int32_t(p10dc_pow3(uint32_t(q - 1)));
        uint32_t center = p10dc_high_center_after_pair(dest_center, RR, p);
        p10dc_add_high_key(z, uint32_t(int32_t(base) + delta), center, fixed_hs);
    }

    uint32_t depth = uint32_t(p10dc_payload_depth(payload));
    if (depth) {
        int32_t delta = p10dc_high_pair_delta(LL, p);
        uint32_t center = p10dc_high_center_after_pair(dest_center, LL, p);
        p10dc_set_high_cross_key(z, uint32_t(int32_t(base) + delta), center, fixed_hs + 2, depth);
    }
    return z;
}

__device__ __forceinline__ BkczPlan p10dc_forward_low_delta(
    uint32_t payload, uint32_t loc, const BucketPhysicalBlock& db, int p
) {
    uint32_t dc = bkci_low_code(loc, db.hs);
    MateID d = p == 1 ? (MateID(dc) | (MateID(db.c) << (2 * LOW_LUT_K))) : minsert(MateID(dc), p, N);
    return p10dc_build_low_plan_delta(d, db.he, p, payload);
}

__device__ __forceinline__ BkczPlan p10dc_forward_high_delta(
    uint32_t payload, uint32_t loc, const BucketPhysicalBlock& db, int p
) {
    uint32_t dc = bkci_high_code(loc, db.he);
    int rel = p - LOW_LUT_K;
    MateID d = minsert(MateID(dc), rel, N);
    return p10dc_build_high_plan_delta(d, db.hs, rel, payload);
}

__device__ __forceinline__ BkczPlan p10dc_reverse_low_delta(
    uint32_t payload, uint32_t loc, const BucketPhysicalBlock& db, int p
) {
    uint32_t dc = bkci_low_code(loc, db.hs);
    MateID d = blocked_exclude_reverse(MateID(dc), LOW_LUT_K + 1, p);
    return p10dc_build_low_plan_delta(d, db.he, p, payload);
}

__device__ __forceinline__ BkczPlan p10dc_reverse_high_delta(
    uint32_t payload, uint32_t loc, const BucketPhysicalBlock& db, int p, bool edge
) {
    uint32_t dc = bkci_high_code(loc, db.he);
    int rel = p - LOW_LUT_K;
    MateID d = edge ? (MateID(db.c) | (MateID(dc) << 2)) : blocked_exclude_reverse(MateID(dc), HIGH_LUT_K + 1, rel);
    return p10dc_build_high_plan_delta(d, db.hs, rel, payload);
}
