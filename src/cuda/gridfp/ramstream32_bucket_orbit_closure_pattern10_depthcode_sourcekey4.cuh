#pragma once

#include "ramstream32_bucket_onepass_pattern10_depthcode.hpp"
#include "ramstream32_bucket_closure_zero_plan.cuh"

// Depthcode-only A/B path for closure-plan source lookup. The canonical
// bkcz_{low,high}_source_ref() still converts each candidate factor code with
// the serial gdx_ternary_key(). A depthcode payload can generate up to
// BKCZ_MAX_LOCAL ordinary candidates plus one CROSS candidate per orbit, so
// those repeated conversions are a visible setup cost even after topology has
// been predecoded. Reuse the already-proven four-symbol ternary conversion
// here without changing any source validity, height, owner, or block checks.

__device__ __forceinline__ bool p10dc_low_source_ref_key4(
    MateID partial, int fixed_he, uint32_t& loc, uint32_t& bid
) {
    constexpr uint64_t MASK = (uint64_t(1) << (2 * LOW_LUT_K)) - 1ull;
    uint32_t c = uint32_t(mget(partial, LOW_LUT_K));
    uint32_t lc = uint32_t(partial & MASK);
    int hs = fixed_he + bkci_delta(c);
    if (fixed_he < 0 || fixed_he > HIGH_LUT_K + 1 || hs < 0 || hs > LOW_LUT_K + 1)
        return false;
    uint32_t z = D_BKF_LOW_DIRECT[bkcz_ternary_key4_legal<LOW_LUT_K>(lc)];
    if (z == BKF_DIRECT_INVALID || int(bkf_direct_height(z)) != hs) return false;
    loc = bkf_direct_locator(z);
    bid = uint32_t(3 * fixed_he + int(c));
    return bid < D_BKF_MAIN_NBLOCKS && bkf_low_main(bkf_loc_owner(loc), bid).valid;
}

__device__ __forceinline__ bool p10dc_high_source_ref_key4(
    MateID partial, int fixed_hs, uint32_t& loc, uint32_t& bid
) {
    constexpr uint64_t MASK = (uint64_t(1) << (2 * HIGH_LUT_K)) - 1ull;
    uint32_t c = uint32_t(mget(partial, 0));
    uint32_t hc = uint32_t((partial >> 2) & MASK);
    int he = fixed_hs - bkci_delta(c);
    if (fixed_hs < 0 || fixed_hs > LOW_LUT_K + 1 || he < 0 || he > HIGH_LUT_K + 1)
        return false;
    uint32_t z = D_BKF_HIGH_DIRECT[bkcz_ternary_key4_legal<HIGH_LUT_K>(hc)];
    if (z == BKF_DIRECT_INVALID || int(bkf_direct_height(z)) != he) return false;
    loc = bkf_direct_locator(z);
    bid = uint32_t(3 * he + int(c));
    return bid < D_BKF_MAIN_NBLOCKS && bkf_high_main(bkf_loc_owner(loc), bid).valid;
}

__device__ __forceinline__ void p10dc_plan_add_low_key4(
    BkczPlan& p, MateID x, int fixed_he
) {
    uint32_t loc = 0, bid = 0;
    if (!p10dc_low_source_ref_key4(x, fixed_he, loc, bid)) return;
    uint32_t n = bkcz_plan_local_n(p);
    if (n >= BKCZ_MAX_LOCAL) return;
    p.local[n] = bkcz_src_pack_device(bid, loc);
    p.meta = (p.meta & ~BKCZ_META_LOCAL_N_MASK) | ((n + 1u) << BKCZ_META_LOCAL_N_SHIFT);
}

__device__ __forceinline__ void p10dc_plan_add_high_key4(
    BkczPlan& p, MateID x, int fixed_hs
) {
    uint32_t loc = 0, bid = 0;
    if (!p10dc_high_source_ref_key4(x, fixed_hs, loc, bid)) return;
    uint32_t n = bkcz_plan_local_n(p);
    if (n >= BKCZ_MAX_LOCAL) return;
    p.local[n] = bkcz_src_pack_device(bid, loc);
    p.meta = (p.meta & ~BKCZ_META_LOCAL_N_MASK) | ((n + 1u) << BKCZ_META_LOCAL_N_SHIFT);
}

// The includes above are intentionally completed before these substitutions.
// The original depthcode implementation is then compiled unchanged except for
// the four source-reference call sites used while materializing BkczPlan.
#define bkcz_plan_add_low p10dc_plan_add_low_key4
#define bkcz_plan_add_high p10dc_plan_add_high_key4
#define bkcz_low_source_ref p10dc_low_source_ref_key4
#define bkcz_high_source_ref p10dc_high_source_ref_key4
#include "ramstream32_bucket_orbit_closure_pattern10_depthcode.cuh"
#undef bkcz_high_source_ref
#undef bkcz_low_source_ref
#undef bkcz_plan_add_high
#undef bkcz_plan_add_low
