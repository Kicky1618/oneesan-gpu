#pragma once

#include "gridfp_transition.hpp"

#include <cstdint>

#if defined(__CUDACC__)
#define ONEESAN_CINV_HD __host__ __device__ __forceinline__
#else
#define ONEESAN_CINV_HD inline
#endif

namespace oneesan::gridfp {

// Enumerate ordinary (same active-side) closure preimages of a post-closure
// partial frontier containing the active half plus the center symbol.  `p` is
// the pair's upper position in this partial coordinate system, so the pair is
// (p-1,p), exactly as mpair().  The post-closure pair must be NN.
//
// RL is the direct preimage. LL/RR additionally undo the remote mate relabel.
// For LL, scanning left from p-2, a destination L is a valid remote candidate
// whenever the intervening suffix is balanced and never went negative.  For
// RR the symmetric statement holds scanning right with destination R.
//
// The routine enumerates topology candidates only. Callers still filter by
// their half-code legality/rank table because an isolated partial frontier need
// not satisfy the full factor boundary height.
template<int MAX_OUT>
ONEESAN_CINV_HD int ordinary_closure_preimages_partial(
    MateID dest, int len, int p, MateID (&out)[MAX_OUT]
) {
    if (p <= 0 || p >= len || mpair(dest, p) != NN) return 0;
    int n = 0;
    auto emit = [&](MateID x) {
        if (n < MAX_OUT) out[n++] = x;
    };

    emit(msetpair(dest, p, RL));

    int bal = 0;
    for (int q = p - 2; q >= 0; --q) {
        MateValue v = mget(dest, q);
        if (bal == 0 && v == L) {
            MateID x = msetpair(dest, p, LL);
            x = mset(x, q, R);
            emit(x);
        }
        if (v == L) ++bal;
        else if (v == R) --bal;
        if (bal < 0) break;
    }

    bal = 0;
    for (int q = p + 1; q < len; ++q) {
        MateValue v = mget(dest, q);
        if (bal == 0 && v == R) {
            MateID x = msetpair(dest, p, RR);
            x = mset(x, q, L);
            emit(x);
        }
        if (v == R) ++bal;
        else if (v == L) --bal;
        if (bal < 0) break;
    }
    return n;
}

} // namespace oneesan::gridfp

#undef ONEESAN_CINV_HD
