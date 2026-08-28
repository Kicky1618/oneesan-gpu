#pragma once

#include "gridfp_reduced_production_discovery_device.cuh"

namespace oneesan::gridfp::reducedprod {

#ifndef RP_RUNTIME_PACK_SHARED_KEYS
#define RP_RUNTIME_PACK_SHARED_KEYS 1
#endif
#ifndef RP_RUNTIME_FAST_DISCOVERY_VALIDITY
#define RP_RUNTIME_FAST_DISCOVERY_VALIDITY 1
#endif
static_assert(RP_RUNTIME_PACK_SHARED_KEYS == 0 || RP_RUNTIME_PACK_SHARED_KEYS == 1,
              "RP_RUNTIME_PACK_SHARED_KEYS must be 0 or 1");
static_assert(RP_RUNTIME_FAST_DISCOVERY_VALIDITY == 0 ||
              RP_RUNTIME_FAST_DISCOVERY_VALIDITY == 1,
              "RP_RUNTIME_FAST_DISCOVERY_VALIDITY must be 0 or 1");

static constexpr std::uint64_t RP_RUNTIME_SHARED_BLOCKED_BIT = 1ULL << 63;
static constexpr std::uint64_t RP_RUNTIME_SHARED_MATE_MASK =
    RP_RUNTIME_SHARED_BLOCKED_BIT - 1ULL;
static_assert(2 * RP_MAX_W < 63,
              "packed shared key requires one free high bit");

#if RP_RUNTIME_PACK_SHARED_KEYS
using RuntimeSharedKey = std::uint64_t;
#else
using RuntimeSharedKey = DeviceKey;
#endif

__device__ __forceinline__ RuntimeSharedKey runtime_shared_key_encode(DeviceKey k) {
#if RP_RUNTIME_PACK_SHARED_KEYS
    return std::uint64_t(k.mate) |
           (k.blocked ? RP_RUNTIME_SHARED_BLOCKED_BIT : 0ULL);
#else
    return k;
#endif
}

__device__ __forceinline__ DeviceKey runtime_shared_key_decode(RuntimeSharedKey k) {
#if RP_RUNTIME_PACK_SHARED_KEYS
    return DeviceKey{
        MateID(std::uint64_t(k) & RP_RUNTIME_SHARED_MATE_MASK),
        std::uint8_t((std::uint64_t(k) & RP_RUNTIME_SHARED_BLOCKED_BIT) != 0)};
#else
    return k;
#endif
}

__device__ __forceinline__ int runtime_find_shared_key(
    const RuntimeSharedKey* a, int n, DeviceKey k
) {
#if RP_RUNTIME_PACK_SHARED_KEYS
    const RuntimeSharedKey needle = runtime_shared_key_encode(k);
    for (int i = 0; i < n; ++i)
        if (a[i] == needle) return i;
#else
    for (int i = 0; i < n; ++i)
        if (key_equal(a[i], k)) return i;
#endif
    return -1;
}

struct RuntimeSharedKeySetSink {
    RuntimeSharedKey* out = nullptr;
    int* n = nullptr;
    int cap = 0;
    __device__ __forceinline__ bool emit(DeviceKey k) {
        if (runtime_find_shared_key(out, *n, k) >= 0) return true;
        if (*n >= cap) return false;
        out[(*n)++] = runtime_shared_key_encode(k);
        return true;
    }
};

struct RuntimeSharedMirroredKeySetSink {
    RuntimeSharedKeySetSink sink{};
    int W = 0;
    __device__ __forceinline__ bool emit(DeviceKey k) {
        return sink.emit(mirror_key_device(k, W));
    }
};

enum RuntimeDiscoveryValidityHint : std::uint8_t {
    RP_RUNTIME_DISCOVERY_CHECK_FULL = 0,
    RP_RUNTIME_DISCOVERY_KNOWN_VALID = 1,
    RP_RUNTIME_DISCOVERY_RL_FROM_VALID_NN = 2,
};

__device__ __forceinline__ bool runtime_discovery_rl_candidate_valid(
    MateID x, int W, int p
) {
#if RP_RUNTIME_FAST_DISCOVERY_VALIDITY
    const int first_high = p + 1;
    const MateID prefix = first_high >= W ? 0 : (x >> (2 * first_high));
    constexpr MateID EVEN = 0x5555555555555555ULL;
    const int r = __popcll(prefix & EVEN);
    const int l = __popcll((prefix >> 1) & EVEN);
    return 1 + l - r > 0;
#else
    return valid_mate_device(x, W);
#endif
}

__device__ __forceinline__ bool runtime_discovery_candidate_valid(
    MateID x, int W, int p, RuntimeDiscoveryValidityHint hint
) {
#if RP_RUNTIME_FAST_DISCOVERY_VALIDITY
    if (hint == RP_RUNTIME_DISCOVERY_KNOWN_VALID) return true;
    if (hint == RP_RUNTIME_DISCOVERY_RL_FROM_VALID_NN)
        return runtime_discovery_rl_candidate_valid(x, W, p);
#endif
    return valid_mate_device(x, W);
}

template<class Sink>
__device__ __forceinline__ bool runtime_discover_blocked_include_candidate_forward(
    MateID x, MateID blocked_dest, int W, int p,
    RuntimeDiscoveryValidityHint hint, Sink& sink
) {
#if RP_RUNTIME_FAST_DISCOVERY_VALIDITY
    if (hint == RP_RUNTIME_DISCOVERY_CHECK_FULL) {
        const IncludeResult z = include_horizontal(x, W, p);
        if (!z.valid || !z.blocked || z.mate != blocked_dest) return true;
        if (!valid_mate_device(x, W)) return true;
        return sink.emit(DeviceKey{x, 0});
    }
#endif
    if (!runtime_discovery_candidate_valid(x, W, p, hint)) return true;
    const IncludeResult z = include_horizontal(x, W, p);
    if (!z.valid || !z.blocked || z.mate != blocked_dest) return true;
    return sink.emit(DeviceKey{x, 0});
}

template<class Sink>
__device__ __forceinline__ bool runtime_discover_blocked_include_preimages_forward(
    MateID b, int W, int p, Sink& sink
) {
    if (is_endpoint(mget(b, p - 1))) {
        const MateID x = minsert(b, p, N);
#if RP_RUNTIME_FAST_DISCOVERY_VALIDITY
        // Inserting N preserves the valid path. RN/LN is exactly the inverse
        // of the blocked NR/NL include branch, so no include recheck is needed.
        if (!sink.emit(DeviceKey{x, 0})) return false;
#else
        if (!runtime_discover_blocked_include_candidate_forward(
                x, b, W, p, RP_RUNTIME_DISCOVERY_KNOWN_VALID, sink))
            return false;
#endif
    }

    const MateID d = minsert(b, p - 1, N);
    if (p <= 0 || p >= W || mpair(d, p) != NN) return true;

    const MateID rl = msetpair(d, p, RL);
#if RP_RUNTIME_FAST_DISCOVERY_VALIDITY
    // RL->NN then shrinking the inserted N is exactly b. Only the leading R
    // can violate the ballot condition, tested by the packed prefix popcounts.
    if (runtime_discovery_rl_candidate_valid(rl, W, p)) {
        if (!sink.emit(DeviceKey{rl, 0})) return false;
    }
#else
    if (!runtime_discover_blocked_include_candidate_forward(
            rl, b, W, p, RP_RUNTIME_DISCOVERY_RL_FROM_VALID_NN, sink))
        return false;
#endif

    int bal = 0;
    for (int q = p - 2; q >= 0; --q) {
        const MateValue v = mget(d, q);
        if (bal == 0 && v == L) {
            MateID x = msetpair(d, p, LL);
            x = mset(x, q, R);
#if RP_RUNTIME_FAST_DISCOVERY_VALIDITY
            // NN->LL raises every affected prefix by two until q; L->R restores
            // the original height there. The balance condition makes q exactly
            // the mate found by include_horizontal(LL), hence the result is b.
            if (!sink.emit(DeviceKey{x, 0})) return false;
#else
            if (!runtime_discover_blocked_include_candidate_forward(
                    x, b, W, p, RP_RUNTIME_DISCOVERY_CHECK_FULL, sink))
                return false;
#endif
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
#if RP_RUNTIME_FAST_DISCOVERY_VALIDITY
            // Symmetrically R->L raises prefixes until RR removes two units;
            // q is exactly the mate found by include_horizontal(RR).
            if (!sink.emit(DeviceKey{x, 0})) return false;
#else
            if (!runtime_discover_blocked_include_candidate_forward(
                    x, b, W, p, RP_RUNTIME_DISCOVERY_CHECK_FULL, sink))
                return false;
#endif
        }
        if (v == R) ++bal;
        else if (v == L) --bal;
        if (bal < 0) break;
    }
    return true;
}

template<class Sink>
__device__ __forceinline__ bool runtime_discover_try_main_inverse_forward(
    MateID x, MateID dest, int W, int p, Sink& sink
) {
    if (!valid_mate_device(x, W)) return true;
    const IncludeResult z = include_horizontal(x, W, p);
    if (z.valid && !z.blocked && z.mate == dest)
        return sink.emit(DeviceKey{x, 0});
    return true;
}

template<class Sink>
__device__ __forceinline__ bool runtime_discover_inverse_reduced_forward(
    DeviceKey dest, int W, int p, Sink& sink
) {
    if (dest.blocked)
        return runtime_discover_blocked_include_preimages_forward(
            dest.mate, W, p, sink);

    const MateID d = dest.mate;
    if (!sink.emit(DeviceKey{d, 0})) return false;

    const MateValuePair w = mpair(d, p);
#if RP_RUNTIME_FAST_DISCOVERY_VALIDITY
    // These three rewrites are exact inverse pairs of the main include cases on
    // a valid destination. Their candidates are valid and include back to d.
    if (w == LR && !sink.emit(DeviceKey{msetpair(d, p, NN), 0})) return false;
    if (w == NR && !sink.emit(DeviceKey{msetpair(d, p, RN), 0})) return false;
    if (w == NL && !sink.emit(DeviceKey{msetpair(d, p, LN), 0})) return false;
#else
    if (w == LR && !runtime_discover_try_main_inverse_forward(
            msetpair(d, p, NN), d, W, p, sink)) return false;
    if (w == NR && !runtime_discover_try_main_inverse_forward(
            msetpair(d, p, RN), d, W, p, sink)) return false;
    if (w == NL && !runtime_discover_try_main_inverse_forward(
            msetpair(d, p, LN), d, W, p, sink)) return false;
#endif

    if (mget(d, p) == N && is_endpoint(mget(d, p - 1))) {
        const MateID b = mshrink(d, p);
#if RP_RUNTIME_FAST_DISCOVERY_VALIDITY
        if (!sink.emit(DeviceKey{b, 1})) return false;
#else
        if (valid_mate_device(b, W - 1) && mget(b, p - 1) != N &&
            blocked_exclude(b, p) == d) {
            if (!sink.emit(DeviceKey{b, 1})) return false;
        }
#endif
    }

    const int q = p - 1;
    const MateValuePair qp = mpair(d, q);
    if (qp == NN || qp == LR) {
        const MateID nn = qp == NN ? d : msetpair(d, q, NN);
        const MateID b = mshrink(nn, q);
#if RP_RUNTIME_FAST_DISCOVERY_VALIDITY
        if (!runtime_discover_blocked_include_preimages_forward(
                b, W, p, sink)) return false;
#else
        if (valid_mate_device(b, W - 1) && mget(b, q - 1) == N) {
            if (!runtime_discover_blocked_include_preimages_forward(
                    b, W, p, sink)) return false;
        }
#endif
    }
    return true;
}

__device__ __forceinline__ bool runtime_discover_inverse_direction_to_shared(
    DeviceKey dest, int W, int p, bool reverse,
    RuntimeSharedKey* source_set, int& source_count, int capacity
) {
    RuntimeSharedKeySetSink base{source_set, &source_count, capacity};
    if (!reverse)
        return runtime_discover_inverse_reduced_forward(dest, W, p, base);

    RuntimeSharedMirroredKeySetSink mirrored{base, W};
    const DeviceKey md = mirror_key_device(dest, W);
    return runtime_discover_inverse_reduced_forward(md, W, W - p, mirrored);
}

static constexpr int RP_RUNTIME_SHARED_KEY_BYTES = sizeof(RuntimeSharedKey);
static constexpr int RP_RUNTIME_DEVICE_KEY_BYTES = sizeof(DeviceKey);

} // namespace oneesan::gridfp::reducedprod
