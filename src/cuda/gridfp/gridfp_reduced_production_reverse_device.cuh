#pragma once

#include "gridfp_reduced_production_device.cuh"

namespace oneesan::gridfp::reducedprod {

__device__ __forceinline__ DeviceKey mirror_key_device(DeviceKey k, int W) {
    return DeviceKey{mirror_mate(k.mate, k.blocked ? W - 1 : W), k.blocked};
}

// Reverse reduced transfer is exactly the horizontal-reflection conjugate of
// the forward reduced transfer.  Using the conjugacy here keeps the CUDA path
// on one authoritative set of local production rules.
__device__ __forceinline__ int reduced_step_reverse(DeviceKey src, int W, int p, DeviceTerm* out) {
    const int fp = W - p;
    DeviceTerm tmp[RP_MAX_TERMS]{};
    const int n = reduced_step_forward(mirror_key_device(src, W), W, fp, tmp);
    if (n < 0) return n;
    for (int i = 0; i < n; ++i) {
        out[i].key = mirror_key_device(tmp[i].key, W);
        out[i].coef = tmp[i].coef;
    }
    return n;
}

__device__ __forceinline__ int inverse_reduced_reverse(DeviceKey dest, int W, int p, DeviceTerm* out) {
    const int fp = W - p;
    DeviceTerm tmp[RP_MAX_TERMS]{};
    const int n = inverse_reduced_forward(mirror_key_device(dest, W), W, fp, tmp);
    if (n < 0) return n;
    for (int i = 0; i < n; ++i) {
        out[i].key = mirror_key_device(tmp[i].key, W);
        out[i].coef = tmp[i].coef;
    }
    return n;
}

__device__ __forceinline__ DeviceKey reverse_component_seed(MateID label, int W, int p, bool& eligible) {
    eligible = !(mget(label, p - 1) == N && mget(label, p) == N);
    if (!eligible) return {};
    if (mget(label, p - 1) != N) return DeviceKey{label, 1};
    return DeviceKey{blocked_exclude_reverse(label, W, p), 0};
}

} // namespace oneesan::gridfp::reducedprod
