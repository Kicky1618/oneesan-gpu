#pragma once

#ifndef RP_RUNTIME_TURN_DIRECT_HIGH_EXPAND_INVERSE
#define RP_RUNTIME_TURN_DIRECT_HIGH_EXPAND_INVERSE 1
#endif
#ifndef RP_RUNTIME_TURN_DISCOVERY_NONN_SCAN
#define RP_RUNTIME_TURN_DISCOVERY_NONN_SCAN 0
#endif
static_assert(
    RP_RUNTIME_TURN_DIRECT_HIGH_EXPAND_INVERSE == 0 ||
    RP_RUNTIME_TURN_DIRECT_HIGH_EXPAND_INVERSE == 1,
    "RP_RUNTIME_TURN_DIRECT_HIGH_EXPAND_INVERSE must be 0 or 1");

namespace oneesan::gridfp::reducedprod {

template<class Sink>
__device__ __forceinline__ bool runtime_turn_discover_expand_high_blocked_direct(
    MateID b, int W, Sink& sink
) {
    const int p = W - 1;

    // Inserting N beyond the high boundary leaves validity unchanged.
    if (is_endpoint(mget(b, p - 1))) {
        if (!sink.emit(DeviceKey{minsert(b, p, N), 0})) return false;
    }

    // At p=W-1 there is no right closure family. If the inserted high pair is
    // NN, the RL seed is automatically valid and only the left LL family can
    // produce additional main preimages.
    const MateID d = minsert(b, p - 1, N);
    if (mpair(d, p) != NN) return true;
    if (!sink.emit(DeviceKey{msetpair(d, p, RL), 0})) return false;

    int bal = 0;
#if RP_RUNTIME_TURN_DISCOVERY_NONN_SCAN
    const std::uint32_t limit =
        p <= 1 ? 0u : ((std::uint32_t(1) << (p - 1)) - 1u);
    std::uint32_t mask = mate_non_n_mask(d, W) & limit;
    while (mask) {
        const int q = mate_msb_index32(mask);
        const MateValue v = mget(d, q);
        if (bal == 0 && v == L) {
            MateID x = msetpair(d, p, LL);
            x = mset(x, q, R);
            if (!sink.emit(DeviceKey{x, 0})) return false;
        }
        if (v == L) ++bal;
        else if (v == R) --bal;
        if (bal < 0) break;
        mask ^= std::uint32_t(1) << q;
    }
#else
    for (int q = p - 2; q >= 0; --q) {
        const MateValue v = mget(d, q);
        if (bal == 0 && v == L) {
            MateID x = msetpair(d, p, LL);
            x = mset(x, q, R);
            if (!sink.emit(DeviceKey{x, 0})) return false;
        }
        if (v == L) ++bal;
        else if (v == R) --bal;
        if (bal < 0) break;
    }
#endif
    return true;
}

template<class Sink>
__device__ __forceinline__ bool runtime_turn_discover_expand_high_direct(
    DeviceKey dest, int W, Sink& sink
) {
    if (dest.blocked)
        return runtime_turn_discover_expand_high_blocked_direct(
            dest.mate, W, sink);

    const MateID d = dest.mate;
    const int p = W - 1;
    if (!sink.emit(DeviceKey{d, 0})) return false;

    const MateValuePair pair = mpair(d, p);
    if (pair == LR && !sink.emit(DeviceKey{msetpair(d, p, NN), 0})) return false;
    if (pair == NR && !sink.emit(DeviceKey{msetpair(d, p, RN), 0})) return false;
    if (pair == NL && !sink.emit(DeviceKey{msetpair(d, p, LN), 0})) return false;

    // The ordinary blocked predecessor branch is intentionally absent: turn
    // expansion source components are main-only. The Q_{W-2} reconstruction
    // below can still generate main preimages of a projected blocked branch.
    const int q = p - 1;
    const MateValuePair qp = mpair(d, q);
    if (qp == NN || qp == LR) {
        const MateID nn = qp == NN ? d : msetpair(d, q, NN);
        const MateID b = mshrink(nn, q);
        // Reachable Q_{W-2} main destinations make the lookahead-N condition
        // and validity structural; both are proved by the host gate.
        if (!runtime_turn_discover_expand_high_blocked_direct(b, W, sink))
            return false;
    }
    return true;
}

// runtime_turn.cuh already routes high expansion through the optimized runtime
// forward-discovery function when RP_RUNTIME_FAST_DISCOVERY_VALIDITY=1.  Wrap
// that call here so the boundary p=W-1 case uses the smaller turn-specific
// inverse while every other p keeps the generic structural implementation.
template<class Sink>
__device__ __forceinline__ bool runtime_turn_discover_forward_dispatch(
    DeviceKey dest, int W, int p, Sink& sink
) {
#if RP_RUNTIME_TURN_DIRECT_HIGH_EXPAND_INVERSE
    if (p == W - 1)
        return runtime_turn_discover_expand_high_direct(dest, W, sink);
#endif
    return runtime_discover_inverse_reduced_forward(dest, W, p, sink);
}

} // namespace oneesan::gridfp::reducedprod

#ifndef RP_RUNTIME_TURN_FORWARD_DISCOVERY_DISPATCH_ACTIVE
#define RP_RUNTIME_TURN_FORWARD_DISCOVERY_DISPATCH_ACTIVE 1
#define runtime_discover_inverse_reduced_forward(...) \
    runtime_turn_discover_forward_dispatch(__VA_ARGS__)
#endif
