#pragma once

#include "gridfp_reduced_production_reverse_device.cuh"

namespace oneesan::gridfp::reducedprod {

#ifndef RP_RUNTIME_DIRECT_REVERSE_SMALL_STEP
#define RP_RUNTIME_DIRECT_REVERSE_SMALL_STEP 1
#endif
#ifndef RP_RUNTIME_DIRECT_REVERSE_P1_MAIN_STEP
#define RP_RUNTIME_DIRECT_REVERSE_P1_MAIN_STEP 1
#endif
#ifndef RP_RUNTIME_DIRECT_FORWARD_LAST_MAIN_STEP
#define RP_RUNTIME_DIRECT_FORWARD_LAST_MAIN_STEP 1
#endif
static_assert(
    RP_RUNTIME_DIRECT_REVERSE_SMALL_STEP == 0 ||
    RP_RUNTIME_DIRECT_REVERSE_SMALL_STEP == 1,
    "RP_RUNTIME_DIRECT_REVERSE_SMALL_STEP must be 0 or 1");
static_assert(
    RP_RUNTIME_DIRECT_REVERSE_P1_MAIN_STEP == 0 ||
    RP_RUNTIME_DIRECT_REVERSE_P1_MAIN_STEP == 1,
    "RP_RUNTIME_DIRECT_REVERSE_P1_MAIN_STEP must be 0 or 1");
static_assert(
    RP_RUNTIME_DIRECT_FORWARD_LAST_MAIN_STEP == 0 ||
    RP_RUNTIME_DIRECT_FORWARD_LAST_MAIN_STEP == 1,
    "RP_RUNTIME_DIRECT_FORWARD_LAST_MAIN_STEP must be 0 or 1");

struct RuntimeSmallTerms {
    DeviceTerm v[3]{};
    int n = 0;
};

__device__ __forceinline__ bool runtime_small_add(
    RuntimeSmallTerms& z, DeviceKey k, int c
) {
    if (!c) return true;
    for (int i = 0; i < z.n; ++i) {
        if (!key_equal(z.v[i].key, k)) continue;
        const int x = int(z.v[i].coef) + c;
        z.v[i].coef = static_cast<std::int8_t>(x);
        if (!x) z.v[i] = z.v[--z.n];
        return true;
    }
    if (z.n >= 3) return false;
    z.v[z.n++] = DeviceTerm{k, static_cast<std::int8_t>(c)};
    return true;
}

__device__ __forceinline__ bool runtime_project_forward(
    DeviceKey k, int W, int q, RuntimeSmallTerms& z
) {
    if (!k.blocked || mget(k.mate, q - 1) != N)
        return runtime_small_add(z, k, 1);
    const MateID nn = blocked_exclude(k.mate, q);
    return runtime_small_add(z, DeviceKey{nn, 0}, 1) &&
           runtime_small_add(z, DeviceKey{msetpair(nn, q, LR), 0}, -1);
}

__device__ __forceinline__ bool runtime_project_reverse(
    DeviceKey k, int W, int q, RuntimeSmallTerms& z
) {
    if (!k.blocked || mget(k.mate, q - 1) != N)
        return runtime_small_add(z, k, 1);
    const MateID nn = blocked_exclude_reverse(k.mate, W, q);
    return runtime_small_add(z, DeviceKey{nn, 0}, 1) &&
           runtime_small_add(z, DeviceKey{msetpair(nn, q, LR), 0}, -1);
}

__device__ __forceinline__ bool runtime_reverse_p1_project_q2_blocked_direct(
    MateID blocked_mate, int W, RuntimeSmallTerms& z
) {
    if (mget(blocked_mate, 1) != N)
        return runtime_small_add(z, DeviceKey{blocked_mate, 1}, 1);
    const MateID nn = blocked_exclude_reverse(blocked_mate, W, 2);
    return runtime_small_add(z, DeviceKey{nn, 0}, 1) &&
           runtime_small_add(z, DeviceKey{msetpair(nn, 2, LR), 0}, -1);
}

__device__ __forceinline__ bool runtime_forward_last_project_blocked_direct(
    MateID blocked_mate, int W, RuntimeSmallTerms& z
) {
    const int q = W - 2;
    if (mget(blocked_mate, q - 1) != N)
        return runtime_small_add(z, DeviceKey{blocked_mate, 1}, 1);
    const MateID nn = blocked_exclude(blocked_mate, q);
    return runtime_small_add(z, DeviceKey{nn, 0}, 1) &&
           runtime_small_add(z, DeviceKey{msetpair(nn, q, LR), 0}, -1);
}

__device__ __forceinline__ bool runtime_small_step_reverse_p1_main_direct(
    MateID mate, int W, RuntimeSmallTerms& z
) {
    const DeviceKey src{mate, 0};
    if (!runtime_small_add(z, src, 1)) return false;
    switch (mpair(mate, 1)) {
    case NN:
        return runtime_small_add(z, DeviceKey{msetpair(mate, 1, LR), 0}, 1);
    case NL:
        return runtime_small_add(z, DeviceKey{msetpair(mate, 1, LN), 0}, 1);
    case NR:
        return runtime_small_add(z, DeviceKey{msetpair(mate, 1, RN), 0}, 1);
    case LN: case RN:
        return runtime_reverse_p1_project_q2_blocked_direct(
            mshrink(mate, 0), W, z);
    case RR: {
        MateID t = msetpair(mate, 1, NN);
        const int q = closure_match_right(t, W, 1);
        if (q < 0) return true;
        t = mset(t, q, R);
        return runtime_reverse_p1_project_q2_blocked_direct(
            mshrink(t, 1), W, z);
    }
    case RL:
        return runtime_reverse_p1_project_q2_blocked_direct(
            mshrink(msetpair(mate, 1, NN), 1), W, z);
    default:
        return true;
    }
}

__device__ __forceinline__ bool runtime_small_step_forward_last_main_direct(
    MateID mate, int W, RuntimeSmallTerms& z
) {
    const DeviceKey src{mate, 0};
    if (!runtime_small_add(z, src, 1)) return false;
    const int p = W - 1;
    switch (mpair(mate, p)) {
    case NN:
        return runtime_small_add(z, DeviceKey{msetpair(mate, p, LR), 0}, 1);
    case NR: case NL:
        return runtime_forward_last_project_blocked_direct(
            mshrink(mate, p), W, z);
    case RN:
        return runtime_small_add(z, DeviceKey{msetpair(mate, p, NR), 0}, 1);
    case LN:
        return runtime_small_add(z, DeviceKey{msetpair(mate, p, NL), 0}, 1);
    case LL: {
        MateID t = msetpair(mate, p, NN);
        const int q = closure_match_left(t, p);
        if (q < 0) return true;
        t = mset(t, q, L);
        return runtime_forward_last_project_blocked_direct(
            mshrink(t, p - 1), W, z);
    }
    case RL:
        return runtime_forward_last_project_blocked_direct(
            mshrink(msetpair(mate, p, NN), p - 1), W, z);
    default:
        return true;
    }
}

__device__ __forceinline__ bool runtime_small_step_forward(
    DeviceKey src, int W, int p, RuntimeSmallTerms& z
) {
    if (!src.blocked) {
#if RP_RUNTIME_DIRECT_FORWARD_LAST_MAIN_STEP
        if (W >= 3 && p == W - 1)
            return runtime_small_step_forward_last_main_direct(src.mate, W, z);
#endif
        if (!runtime_small_add(z, src, 1)) return false;
        const IncludeResult x = include_horizontal(src.mate, W, p);
        if (!x.valid) return true;
        return runtime_project_forward(
            DeviceKey{x.mate, std::uint8_t(x.blocked)}, W, p - 1, z);
    }
    return runtime_small_add(
        z, DeviceKey{blocked_exclude(src.mate, p), 0}, 1);
}

__device__ __forceinline__ bool runtime_small_step_reverse_direct(
    DeviceKey src, int W, int p, RuntimeSmallTerms& z
) {
    if (!src.blocked) {
#if RP_RUNTIME_DIRECT_REVERSE_P1_MAIN_STEP
        if (p == 1)
            return runtime_small_step_reverse_p1_main_direct(src.mate, W, z);
#endif
        if (!runtime_small_add(z, src, 1)) return false;
        const IncludeResult x = include_horizontal_reverse(src.mate, W, p);
        if (!x.valid) return true;
        return runtime_project_reverse(
            DeviceKey{x.mate, std::uint8_t(x.blocked)}, W, p + 1, z);
    }
    return runtime_small_add(
        z, DeviceKey{blocked_exclude_reverse(src.mate, W, p), 0}, 1);
}

__device__ __forceinline__ bool runtime_small_step(
    DeviceKey src, int W, int p, bool reverse, RuntimeSmallTerms& z
) {
    if (!reverse) return runtime_small_step_forward(src, W, p, z);
#if RP_RUNTIME_DIRECT_REVERSE_SMALL_STEP
    return runtime_small_step_reverse_direct(src, W, p, z);
#else
    const int fp = W - p;
    RuntimeSmallTerms tmp;
    if (!runtime_small_step_forward(mirror_key_device(src, W), W, fp, tmp))
        return false;
    for (int i = 0; i < tmp.n; ++i) {
        if (!runtime_small_add(
                z, mirror_key_device(tmp.v[i].key, W), tmp.v[i].coef))
            return false;
    }
    return true;
#endif
}

__device__ __forceinline__ DeviceKey runtime_component_seed(
    MateID label, int W, int p, bool reverse, bool& eligible
) {
    return reverse ? reverse_component_seed(label, W, p, eligible)
                   : forward_component_seed(label, W, p, eligible);
}

__device__ __forceinline__ int runtime_find_key(
    const DeviceKey* a, int n, DeviceKey k
) {
    for (int i = 0; i < n; ++i)
        if (key_equal(a[i], k)) return i;
    return -1;
}

__device__ __forceinline__ void runtime_set_error(int* error, int code) {
    if (error) atomicCAS(error, 0, code);
}

} // namespace oneesan::gridfp::reducedprod
