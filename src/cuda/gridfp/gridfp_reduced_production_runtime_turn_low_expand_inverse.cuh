#pragma once

#ifndef RP_RUNTIME_TURN_DIRECT_LOW_EXPAND_INVERSE
#define RP_RUNTIME_TURN_DIRECT_LOW_EXPAND_INVERSE 1
#endif
#ifndef RP_RUNTIME_TURN_DISCOVERY_NONN_SCAN
#define RP_RUNTIME_TURN_DISCOVERY_NONN_SCAN 0
#endif
static_assert(
    RP_RUNTIME_TURN_DIRECT_LOW_EXPAND_INVERSE == 0 ||
    RP_RUNTIME_TURN_DIRECT_LOW_EXPAND_INVERSE == 1,
    "RP_RUNTIME_TURN_DIRECT_LOW_EXPAND_INVERSE must be 0 or 1");

namespace oneesan::gridfp::reducedprod {

template<class Sink>
__device__ __forceinline__ bool runtime_turn_discover_expand_low_blocked_direct(
    MateID b, int W, Sink& sink
) {
    // Reflection of the forward p=W-1 deferred endpoint preimage. For a
    // canonical reverse Q_2 blocked destination this source is structurally
    // valid, so no reverse-validity scan or transition recheck is necessary.
    if (is_endpoint(mget(b, 0))) {
        if (!sink.emit(DeviceKey{minsert(b, 0, N), 0})) return false;
    }

    // Reflection of the high-boundary closure reconstruction. In direct
    // reverse coordinates only the RL seed and RR family extend to the right.
    const MateID d = minsert(b, 1, N);
    if (mpair(d, 1) != NN) return true;
    if (!sink.emit(DeviceKey{msetpair(d, 1, RL), 0})) return false;

    int bal = 0;
#if RP_RUNTIME_TURN_DISCOVERY_NONN_SCAN
    std::uint32_t mask = mate_non_n_mask(d, W) & ~std::uint32_t(3u);
    while (mask) {
        const int q = mate_lsb_index32(mask);
        const MateValue v = mget(d, q);
        if (bal == 0 && v == R) {
            MateID x = msetpair(d, 1, RR);
            x = mset(x, q, L);
            if (!sink.emit(DeviceKey{x, 0})) return false;
        }
        if (v == R) ++bal;
        else if (v == L) --bal;
        if (bal < 0) break;
        mask &= mask - 1u;
    }
#else
    for (int q = 2; q < W; ++q) {
        const MateValue v = mget(d, q);
        if (bal == 0 && v == R) {
            MateID x = msetpair(d, 1, RR);
            x = mset(x, q, L);
            if (!sink.emit(DeviceKey{x, 0})) return false;
        }
        if (v == R) ++bal;
        else if (v == L) --bal;
        if (bal < 0) break;
    }
#endif
    return true;
}

template<class Sink>
__device__ __forceinline__ bool runtime_turn_discover_expand_low_direct(
    DeviceKey dest, int W, Sink& sink
) {
    // Expansion sources are main-only. A blocked destination can therefore
    // only be reached through an included main branch.
    if (dest.blocked)
        return runtime_turn_discover_expand_low_blocked_direct(dest.mate, W, sink);

    const MateID d = dest.mate;
    if (!sink.emit(DeviceKey{d, 0})) return false;

    // Reflection of the three ordinary forward inverse rewrites at p=W-1.
    // Reachable reverse-basis destinations make all three structurally valid.
    const MateValuePair pair = mpair(d, 1);
    if (pair == LR && !sink.emit(DeviceKey{msetpair(d, 1, NN), 0})) return false;
    if (pair == LN && !sink.emit(DeviceKey{msetpair(d, 1, NL), 0})) return false;
    if (pair == RN && !sink.emit(DeviceKey{msetpair(d, 1, NR), 0})) return false;

    // Reflection of the quotient-projection reconstruction at forward
    // q=W-2. The reconstructed blocked word is reverse-valid by construction;
    // only the canonical lookahead-N condition remains dynamic.
    if (W > 2) {
        const MateValuePair qp = mpair(d, 2);
        if (qp == NN || qp == LR) {
            const MateID nn = qp == NN ? d : msetpair(d, 2, NN);
            const MateID b = mshrink(nn, 1);
            if (mget(b, 1) == N) {
                if (!runtime_turn_discover_expand_low_blocked_direct(b, W, sink))
                    return false;
            }
        }
    }
    return true;
}

} // namespace oneesan::gridfp::reducedprod

#include "gridfp_reduced_production_runtime_turn_high_expand_inverse.cuh"
