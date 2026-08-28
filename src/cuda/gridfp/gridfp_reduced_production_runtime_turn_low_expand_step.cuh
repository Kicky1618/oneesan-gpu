#pragma once

#ifndef RP_RUNTIME_TURN_DIRECT_LOW_EXPAND_STEP
#define RP_RUNTIME_TURN_DIRECT_LOW_EXPAND_STEP 1
#endif
static_assert(
    RP_RUNTIME_TURN_DIRECT_LOW_EXPAND_STEP == 0 ||
    RP_RUNTIME_TURN_DIRECT_LOW_EXPAND_STEP == 1,
    "RP_RUNTIME_TURN_DIRECT_LOW_EXPAND_STEP must be 0 or 1");

namespace oneesan::gridfp::reducedprod {

__device__ __forceinline__ bool runtime_turn_project_reverse_q2_direct(
    MateID mate, bool blocked, int W, RuntimeSmallTerms& z
) {
    if (!blocked || mget(mate, 1) != N)
        return runtime_small_add(z, DeviceKey{mate, std::uint8_t(blocked)}, 1);
    const MateID nn = blocked_exclude_reverse(mate, W, 2);
    return runtime_small_add(z, DeviceKey{nn, 0}, 1) &&
           runtime_small_add(z, DeviceKey{msetpair(nn, 2, LR), 0}, -1);
}

__device__ __forceinline__ bool runtime_turn_expand_low_step_direct(
    DeviceKey src, int W, RuntimeSmallTerms& z
) {
    if (src.blocked) return false;
    if (!runtime_small_add(z, src, 1)) return false;

    switch (mpair(src.mate, 1)) {
    case NN:
        return runtime_small_add(
            z, DeviceKey{msetpair(src.mate, 1, LR), 0}, 1);
    case NL:
        return runtime_small_add(
            z, DeviceKey{msetpair(src.mate, 1, LN), 0}, 1);
    case NR:
        return runtime_small_add(
            z, DeviceKey{msetpair(src.mate, 1, RN), 0}, 1);
    case LN: case RN:
        return runtime_turn_project_reverse_q2_direct(
            mshrink(src.mate, 0), true, W, z);
    case RR: {
        MateID t = msetpair(src.mate, 1, NN);
        const int q = closure_match_right(t, W, 1);
        if (q < 0) return true;
        t = mset(t, q, R);
        return runtime_turn_project_reverse_q2_direct(
            mshrink(t, 1), true, W, z);
    }
    case RL:
        return runtime_turn_project_reverse_q2_direct(
            mshrink(msetpair(src.mate, 1, NN), 1), true, W, z);
    default:
        return true;
    }
}

} // namespace oneesan::gridfp::reducedprod
