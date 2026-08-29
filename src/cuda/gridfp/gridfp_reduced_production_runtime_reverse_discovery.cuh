#pragma once

#include "gridfp_reduced_production_runtime_shared_key.cuh"

#ifndef RP_RUNTIME_DIRECT_REVERSE_DISCOVERY
#define RP_RUNTIME_DIRECT_REVERSE_DISCOVERY RP_RUNTIME_FAST_DISCOVERY_VALIDITY
#endif
static_assert(RP_RUNTIME_DIRECT_REVERSE_DISCOVERY == 0 ||
              RP_RUNTIME_DIRECT_REVERSE_DISCOVERY == 1,
              "RP_RUNTIME_DIRECT_REVERSE_DISCOVERY must be 0 or 1");

namespace oneesan::gridfp::reducedprod {

template<class Sink>
__device__ __forceinline__ bool runtime_discover_blocked_candidate_reverse_checked(
    MateID x, MateID blocked_dest, int W, int p, Sink& sink
) {
    if (!valid_mate_device(x, W)) return true;
    const IncludeResult z = include_horizontal_reverse(x, W, p);
    if (!z.valid || !z.blocked || z.mate != blocked_dest) return true;
    return sink.emit(DeviceKey{x, 0});
}

template<class Sink>
__device__ __forceinline__ bool runtime_discover_blocked_include_preimages_reverse(
    MateID b, int W, int p, Sink& sink
) {
    if (p <= 0 || p >= W) return false;

    // Reverse deferred LN/RN branch removes the low member of the local pair.
    // Restoring that N does not change path height, so a valid blocked word
    // gives a valid main source whenever the adjacent retained symbol is an
    // endpoint.
    if (is_endpoint(mget(b, p - 1))) {
        const MateID x = minsert(b, p - 1, N);
#if RP_RUNTIME_FAST_DISCOVERY_VALIDITY
        if (!sink.emit(DeviceKey{x, 0})) return false;
#else
        if (!runtime_discover_blocked_candidate_reverse_checked(
                x, b, W, p, sink)) return false;
#endif
    }

    // Reverse LL/RR/RL blocked branches remove original position p. Restore N
    // there and invert the same local closure relation directly in the original
    // coordinate system. LL/RR generated candidates are structurally valid;
    // RL alone needs the usual high-prefix height test.
    const MateID d = minsert(b, p, N);
    if (mpair(d, p) != NN) return true;

    const MateID rl = msetpair(d, p, RL);
#if RP_RUNTIME_FAST_DISCOVERY_VALIDITY
    if (runtime_discovery_rl_candidate_valid(rl, W, p)) {
        if (!sink.emit(DeviceKey{rl, 0})) return false;
    }
#else
    if (!runtime_discover_blocked_candidate_reverse_checked(
            rl, b, W, p, sink)) return false;
#endif

#if RP_RUNTIME_DISCOVERY_ENDPOINT_SCAN
    const std::uint32_t endpoints = runtime_discovery_endpoint_mask(d, W);
    std::uint32_t left = p <= 1 ? 0u :
        (endpoints & ((std::uint32_t(1) << (p - 1)) - 1u));
    int bal = 0;
    while (left) {
        const int q = 31 - __clz(left);
        const MateValue v = mget(d, q);
        if (bal == 0 && v == L) {
            MateID x = msetpair(d, p, LL);
            x = mset(x, q, R);
#if RP_RUNTIME_FAST_DISCOVERY_VALIDITY
            if (!sink.emit(DeviceKey{x, 0})) return false;
#else
            if (!runtime_discover_blocked_candidate_reverse_checked(
                    x, b, W, p, sink)) return false;
#endif
        }
        if (v == L) ++bal;
        else --bal;
        left ^= std::uint32_t(1) << q;
        if (bal < 0) break;
    }

    const std::uint32_t width_mask =
        W == 32 ? ~0u : ((std::uint32_t(1) << W) - 1u);
    const std::uint32_t low_mask =
        p + 1 >= 32 ? ~0u : ((std::uint32_t(1) << (p + 1)) - 1u);
    std::uint32_t right = endpoints & width_mask & ~low_mask;
    bal = 0;
    while (right) {
        const int q = __ffs(right) - 1;
        const MateValue v = mget(d, q);
        if (bal == 0 && v == R) {
            MateID x = msetpair(d, p, RR);
            x = mset(x, q, L);
#if RP_RUNTIME_FAST_DISCOVERY_VALIDITY
            if (!sink.emit(DeviceKey{x, 0})) return false;
#else
            if (!runtime_discover_blocked_candidate_reverse_checked(
                    x, b, W, p, sink)) return false;
#endif
        }
        if (v == R) ++bal;
        else --bal;
        right &= right - 1u;
        if (bal < 0) break;
    }
#else
    int bal = 0;
    for (int q = p - 2; q >= 0; --q) {
        const MateValue v = mget(d, q);
        if (bal == 0 && v == L) {
            MateID x = msetpair(d, p, LL);
            x = mset(x, q, R);
#if RP_RUNTIME_FAST_DISCOVERY_VALIDITY
            if (!sink.emit(DeviceKey{x, 0})) return false;
#else
            if (!runtime_discover_blocked_candidate_reverse_checked(
                    x, b, W, p, sink)) return false;
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
            if (!sink.emit(DeviceKey{x, 0})) return false;
#else
            if (!runtime_discover_blocked_candidate_reverse_checked(
                    x, b, W, p, sink)) return false;
#endif
        }
        if (v == R) ++bal;
        else if (v == L) --bal;
        if (bal < 0) break;
    }
#endif
    return true;
}

template<class Sink>
__device__ __forceinline__ bool runtime_discover_try_main_inverse_reverse(
    MateID x, MateID dest, int W, int p, Sink& sink
) {
    if (!valid_mate_device(x, W)) return true;
    const IncludeResult z = include_horizontal_reverse(x, W, p);
    if (z.valid && !z.blocked && z.mate == dest)
        return sink.emit(DeviceKey{x, 0});
    return true;
}

template<class Sink>
__device__ __forceinline__ bool runtime_discover_inverse_reduced_reverse(
    DeviceKey dest, int W, int p, Sink& sink
) {
    if (p <= 0 || p >= W) return false;
    if (dest.blocked)
        return runtime_discover_blocked_include_preimages_reverse(
            dest.mate, W, p, sink);

    const MateID d = dest.mate;
    if (!sink.emit(DeviceKey{d, 0})) return false;

    // Exact inverses of reverse-coordinate nonblocked local rewrites.
    const MateValuePair w = mpair(d, p);
#if RP_RUNTIME_FAST_DISCOVERY_VALIDITY
    if (w == LR && !sink.emit(DeviceKey{msetpair(d, p, NN), 0})) return false;
    if (w == LN && !sink.emit(DeviceKey{msetpair(d, p, NL), 0})) return false;
    if (w == RN && !sink.emit(DeviceKey{msetpair(d, p, NR), 0})) return false;
#else
    if (w == LR && !runtime_discover_try_main_inverse_reverse(
            msetpair(d, p, NN), d, W, p, sink)) return false;
    if (w == LN && !runtime_discover_try_main_inverse_reverse(
            msetpair(d, p, NL), d, W, p, sink)) return false;
    if (w == RN && !runtime_discover_try_main_inverse_reverse(
            msetpair(d, p, NR), d, W, p, sink)) return false;
#endif

    // Retained blocked source excluded branch: reverse blocked_exclude inserts N
    // at p-1, so removing that N reconstructs the source.
    if (mget(d, p - 1) == N && is_endpoint(mget(d, p))) {
        const MateID b = mshrink(d, p - 1);
#if RP_RUNTIME_FAST_DISCOVERY_VALIDITY
        if (!sink.emit(DeviceKey{b, 1})) return false;
#else
        if (valid_mate_device(b, W - 1) && mget(b, p - 1) != N &&
            blocked_exclude_reverse(b, W, p) == d) {
            if (!sink.emit(DeviceKey{b, 1})) return false;
        }
#endif
    }

    // Q_{p+1} projection reconstruction. A noncanonical blocked result had N at
    // original position p; remove that restored N and enumerate all reverse
    // included main preimages directly.
    const int q = p + 1;
    if (q < W) {
        const MateValuePair qp = mpair(d, q);
        if (qp == NN || qp == LR) {
            const MateID nn = qp == NN ? d : msetpair(d, q, NN);
            const MateID b = mshrink(nn, p);
#if RP_RUNTIME_FAST_DISCOVERY_VALIDITY
            if (!runtime_discover_blocked_include_preimages_reverse(
                    b, W, p, sink)) return false;
#else
            if (valid_mate_device(b, W - 1) && mget(b, p) == N) {
                if (!runtime_discover_blocked_include_preimages_reverse(
                        b, W, p, sink)) return false;
            }
#endif
        }
    }
    return true;
}

__device__ __forceinline__ bool runtime_discover_inverse_direction_to_shared_direct(
    DeviceKey dest, int W, int p, bool reverse,
    RuntimeSharedKey* source_set, int& source_count, int capacity,
    std::uint64_t* source_signature = nullptr
) {
    RuntimeSharedKeySetSink base{
        source_set, &source_count, capacity, source_signature};
    if (!reverse)
        return runtime_discover_inverse_reduced_forward(dest, W, p, base);
#if RP_RUNTIME_DIRECT_REVERSE_DISCOVERY
    return runtime_discover_inverse_reduced_reverse(dest, W, p, base);
#else
    RuntimeSharedMirroredKeySetSink mirrored{base, W};
    const DeviceKey md = mirror_key_device(dest, W);
    return runtime_discover_inverse_reduced_forward(md, W, W - p, mirrored);
#endif
}

} // namespace oneesan::gridfp::reducedprod

#ifndef RP_RUNTIME_REVERSE_DISCOVERY_DIRECTION_DISPATCH_ACTIVE
#define RP_RUNTIME_REVERSE_DISCOVERY_DIRECTION_DISPATCH_ACTIVE 1
#define runtime_discover_inverse_direction_to_shared(...) \
    runtime_discover_inverse_direction_to_shared_direct(__VA_ARGS__)
#endif
