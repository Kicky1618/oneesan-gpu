#pragma once

#include "gridfp_reduced_production_reverse_device.cuh"

namespace oneesan::gridfp::reducedprod {

struct DeviceKeySetSink {
    DeviceKey* out = nullptr;
    int* n = nullptr;
    int cap = 0;

    __device__ __forceinline__ bool emit(DeviceKey k) {
        for (int i = 0; i < *n; ++i)
            if (key_equal(out[i], k)) return true;
        if (*n >= cap) return false;
        out[(*n)++] = k;
        return true;
    }
};

struct MirroredDeviceKeySetSink {
    DeviceKeySetSink sink{};
    int W = 0;

    __device__ __forceinline__ bool emit(DeviceKey k) {
        return sink.emit(mirror_key_device(k, W));
    }
};

template<class Sink>
__device__ __forceinline__ bool discover_blocked_include_candidate_forward(
    MateID x, MateID blocked_dest, int W, int p, Sink& sink
) {
    if (!valid_mate_device(x, W)) return true;
    const IncludeResult z = include_horizontal(x, W, p);
    if (!z.valid || !z.blocked || z.mate != blocked_dest) return true;
    return sink.emit(DeviceKey{x, 0});
}

// Stream the main sources whose included branch produces one blocked state.
// This is the discovery-only counterpart of blocked_include_preimages_forward_device,
// but it never materializes a temporary candidate array.
template<class Sink>
__device__ __forceinline__ bool discover_blocked_include_preimages_forward(
    MateID b, int W, int p, Sink& sink
) {
    if (is_endpoint(mget(b, p - 1))) {
        const MateID x = minsert(b, p, N);
        if (!discover_blocked_include_candidate_forward(x, b, W, p, sink)) return false;
    }

    const MateID d = minsert(b, p - 1, N);
    if (p <= 0 || p >= W || mpair(d, p) != NN) return true;

    if (!discover_blocked_include_candidate_forward(msetpair(d, p, RL), b, W, p, sink))
        return false;

    int bal = 0;
    for (int q = p - 2; q >= 0; --q) {
        const MateValue v = mget(d, q);
        if (bal == 0 && v == L) {
            MateID x = msetpair(d, p, LL);
            x = mset(x, q, R);
            if (!discover_blocked_include_candidate_forward(x, b, W, p, sink)) return false;
        }
        if (v == L) ++bal;
        else if (v == R) --bal;
        if (bal < 0) break;
    }

    bal = 0;
    for (int q = p + 1; q < W; ++q) {
        const MateValue v = mget(d, q);
        if (bal == 0 && v == R) {
            MateID x = msetpair(d, p, RR);
            x = mset(x, q, L);
            if (!discover_blocked_include_candidate_forward(x, b, W, p, sink)) return false;
        }
        if (v == R) ++bal;
        else if (v == L) --bal;
        if (bal < 0) break;
    }
    return true;
}

template<class Sink>
__device__ __forceinline__ bool discover_try_main_inverse_forward(
    MateID x, MateID dest, int W, int p, Sink& sink
) {
    if (!valid_mate_device(x, W)) return true;
    const IncludeResult z = include_horizontal(x, W, p);
    if (z.valid && !z.blocked && z.mate == dest)
        return sink.emit(DeviceKey{x, 0});
    return true;
}

// Enumerate only the support of the inverse column.  Coefficients are not
// needed while reconstructing a connected component, so the signed temporary
// DeviceTerm[RP_MAX_TERMS] buffer can be eliminated entirely.
template<class Sink>
__device__ __forceinline__ bool discover_inverse_reduced_forward(
    DeviceKey dest, int W, int p, Sink& sink
) {
    if (dest.blocked)
        return discover_blocked_include_preimages_forward(dest.mate, W, p, sink);

    const MateID d = dest.mate;
    if (!sink.emit(DeviceKey{d, 0})) return false;

    const MateValuePair w = mpair(d, p);
    if (w == LR && !discover_try_main_inverse_forward(msetpair(d, p, NN), d, W, p, sink)) return false;
    if (w == NR && !discover_try_main_inverse_forward(msetpair(d, p, RN), d, W, p, sink)) return false;
    if (w == NL && !discover_try_main_inverse_forward(msetpair(d, p, LN), d, W, p, sink)) return false;

    if (mget(d, p) == N && is_endpoint(mget(d, p - 1))) {
        const MateID b = mshrink(d, p);
        if (valid_mate_device(b, W - 1) && mget(b, p - 1) != N && blocked_exclude(b, p) == d) {
            if (!sink.emit(DeviceKey{b, 1})) return false;
        }
    }

    const int q = p - 1;
    const MateValuePair qp = mpair(d, q);
    if (qp == NN || qp == LR) {
        const MateID nn = qp == NN ? d : msetpair(d, q, NN);
        const MateID b = mshrink(nn, q);
        if (valid_mate_device(b, W - 1) && mget(b, q - 1) == N) {
            if (!discover_blocked_include_preimages_forward(b, W, p, sink)) return false;
        }
    }
    return true;
}

__device__ __forceinline__ bool discover_inverse_direction_to_set(
    DeviceKey dest,
    int W,
    int p,
    bool reverse,
    DeviceKey* source_set,
    int& source_count,
    int capacity
) {
    DeviceKeySetSink base{source_set, &source_count, capacity};
    if (!reverse)
        return discover_inverse_reduced_forward(dest, W, p, base);

    MirroredDeviceKeySetSink mirrored{base, W};
    const DeviceKey md = mirror_key_device(dest, W);
    return discover_inverse_reduced_forward(md, W, W - p, mirrored);
}

} // namespace oneesan::gridfp::reducedprod
