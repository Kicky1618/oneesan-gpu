#pragma once

#include "gridfp_reduced_production_grouped_device.cuh"
#include "gridfp_reduced_production_reverse_device.cuh"

namespace oneesan::gridfp::reducedprod {

// Q1-forward -> main compression at the physical left edge.
__device__ __forceinline__ int turn_compress_step(
    DeviceKey src, int W, DeviceTerm* out
) {
    int n = 0;
    if (!src.blocked) {
        n = add_term(out, n, src, 1);
        if (n < 0) return n;
        const IncludeResult z = include_horizontal(src.mate, W, 1);
        if (z.valid) {
            if (z.blocked) return -1; // p=1 must close immediately to main.
            n = add_term(out, n, DeviceKey{z.mate, 0}, 1);
        }
        return n;
    }
    return add_term(out, n, DeviceKey{blocked_exclude(src.mate, 1), 0}, 1);
}

__device__ __forceinline__ int turn_try_main_preimage(
    MateID x, MateID dest, int W, DeviceTerm* out, int n
) {
    if (!valid_mate_device(x, W)) return n;
    const IncludeResult z = include_horizontal(x, W, 1);
    if (z.valid && !z.blocked && z.mate == dest)
        return add_term(out, n, DeviceKey{x, 0}, 1);
    return n;
}

// Exact incoming list for the rectangular Q1-forward -> main edge map.
// This is table-free and reuses the ordinary closure inverse at p=1.
__device__ __forceinline__ int turn_compress_inverse(
    DeviceKey dest, int W, DeviceTerm* out
) {
    if (dest.blocked) return -1;
    const MateID d = dest.mate;
    int n = add_term(out, 0, DeviceKey{d, 0}, 1); // excluded identity
    if (n < 0) return n;

    const MateValuePair pair = mpair(d, 1);
    if (pair == LR) n = turn_try_main_preimage(msetpair(d, 1, NN), d, W, out, n);
    if (n < 0) return n;
    if (pair == RN) n = turn_try_main_preimage(msetpair(d, 1, NR), d, W, out, n);
    if (n < 0) return n;
    if (pair == LN) n = turn_try_main_preimage(msetpair(d, 1, NL), d, W, out, n);
    if (n < 0) return n;
    if (pair == NR) n = turn_try_main_preimage(msetpair(d, 1, RN), d, W, out, n);
    if (n < 0) return n;
    if (pair == NL) n = turn_try_main_preimage(msetpair(d, 1, LN), d, W, out, n);
    if (n < 0) return n;

    if (pair == NN) {
        MateID cand[RP_MAX_TERMS]{};
        const int nc = ordinary_closure_preimages_partial(d, W, 1, cand);
        for (int i = 0; i < nc; ++i) {
            n = turn_try_main_preimage(cand[i], d, W, out, n);
            if (n < 0) return n;
        }
    }

    // Blocked Q1 sources have bit 0 occupied and only their excluded branch.
    if (mget(d, 1) == N && is_endpoint(mget(d, 0))) {
        const MateID b = mshrink(d, 1);
        if (valid_mate_device(b, W - 1) && mget(b, 0) != N &&
            blocked_exclude(b, 1) == d) {
            n = add_term(out, n, DeviceKey{b, 1}, 1);
        }
    }
    return n;
}

// There is exactly one Q1-compression component for every width-(W-1)
// Motzkin label.  This seed rule was exhaustively checked through W=12 on CPU.
__device__ __forceinline__ DeviceKey turn_compress_seed(MateID label, int W) {
    if (mget(label, 0) != N) return DeviceKey{label, 1};
    return DeviceKey{blocked_exclude(label, 1), 0};
}

// main -> Q2-reverse expansion is the ordinary reverse reduced p=1 map with
// its source restricted to main states.
__device__ __forceinline__ int turn_expand_step(
    DeviceKey src, int W, DeviceTerm* out
) {
    if (src.blocked) return -1;
    return reduced_step_reverse(src, W, 1, out);
}

__device__ __forceinline__ int turn_expand_inverse(
    DeviceKey dest, int W, DeviceTerm* out
) {
    DeviceTerm tmp[RP_MAX_TERMS]{};
    const int n = inverse_reduced_reverse(dest, W, 1, tmp);
    if (n < 0) return n;
    int z = 0;
    for (int i = 0; i < n; ++i) {
        if (tmp[i].key.blocked || !tmp[i].coef) continue;
        z = add_term(out, z, tmp[i].key, tmp[i].coef);
        if (z < 0) return z;
    }
    return z;
}

// Expansion components have the same label set as a reverse reduced p=1
// component, but a main seed is available for every eligible label.
__device__ __forceinline__ DeviceKey turn_expand_seed(MateID label, int W) {
    return DeviceKey{blocked_exclude_reverse(label, W, 1), 0};
}

// The high-edge turn is exactly the horizontal-reflection conjugate of the
// low-edge turn.  These wrappers keep one authoritative implementation of the
// rectangular edge maps and avoid a second set of closure rules.
__device__ __forceinline__ int turn_compress_step_edge(
    DeviceKey src, int W, bool high, DeviceTerm* out
) {
    if (!high) return turn_compress_step(src, W, out);
    DeviceTerm tmp[RP_MAX_TERMS]{};
    const int n = turn_compress_step(mirror_key_device(src, W), W, tmp);
    if (n < 0) return n;
    for (int i = 0; i < n; ++i) {
        out[i].key = mirror_key_device(tmp[i].key, W);
        out[i].coef = tmp[i].coef;
    }
    return n;
}

__device__ __forceinline__ int turn_compress_inverse_edge(
    DeviceKey dest, int W, bool high, DeviceTerm* out
) {
    if (!high) return turn_compress_inverse(dest, W, out);
    DeviceTerm tmp[RP_MAX_TERMS]{};
    const int n = turn_compress_inverse(mirror_key_device(dest, W), W, tmp);
    if (n < 0) return n;
    for (int i = 0; i < n; ++i) {
        out[i].key = mirror_key_device(tmp[i].key, W);
        out[i].coef = tmp[i].coef;
    }
    return n;
}

__device__ __forceinline__ DeviceKey turn_compress_seed_edge(
    MateID label, int W, bool high
) {
    if (!high) return turn_compress_seed(label, W);
    const MateID low_label = mirror_mate(label, W - 1);
    return mirror_key_device(turn_compress_seed(low_label, W), W);
}

__device__ __forceinline__ int turn_expand_step_edge(
    DeviceKey src, int W, bool high, DeviceTerm* out
) {
    if (!high) return turn_expand_step(src, W, out);
    DeviceTerm tmp[RP_MAX_TERMS]{};
    const int n = turn_expand_step(mirror_key_device(src, W), W, tmp);
    if (n < 0) return n;
    for (int i = 0; i < n; ++i) {
        out[i].key = mirror_key_device(tmp[i].key, W);
        out[i].coef = tmp[i].coef;
    }
    return n;
}

__device__ __forceinline__ int turn_expand_inverse_edge(
    DeviceKey dest, int W, bool high, DeviceTerm* out
) {
    if (!high) return turn_expand_inverse(dest, W, out);
    DeviceTerm tmp[RP_MAX_TERMS]{};
    const int n = turn_expand_inverse(mirror_key_device(dest, W), W, tmp);
    if (n < 0) return n;
    for (int i = 0; i < n; ++i) {
        out[i].key = mirror_key_device(tmp[i].key, W);
        out[i].coef = tmp[i].coef;
    }
    return n;
}

__device__ __forceinline__ DeviceKey turn_expand_seed_edge(
    MateID label, int W, bool high
) {
    if (!high) return turn_expand_seed(label, W);
    const MateID low_label = mirror_mate(label, W - 1);
    return mirror_key_device(turn_expand_seed(low_label, W), W);
}

} // namespace oneesan::gridfp::reducedprod
