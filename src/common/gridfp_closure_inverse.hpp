#pragma once

#include "gridfp_transition.hpp"
#include "gridfp_transition_reverse.hpp"

#include <cstdint>

#if defined(__CUDACC__)
#define ONEESAN_CINV_HD __host__ __device__ __forceinline__
#else
#define ONEESAN_CINV_HD inline
#endif

namespace oneesan::gridfp {

template<int MAX_OUT>
ONEESAN_CINV_HD int ordinary_closure_preimages_partial(
    MateID dest, int len, int p, MateID (&out)[MAX_OUT]
) {
    if (p <= 0 || p >= len || mpair(dest, p) != NN) return 0;
    int n = 0;

    if (n < MAX_OUT) out[n++] = msetpair(dest, p, RL);

    int bal = 0;
    for (int q = p - 2; q >= 0; --q) {
        MateValue v = mget(dest, q);
        if (bal == 0 && v == L) {
            MateID x = msetpair(dest, p, LL);
            x = mset(x, q, R);
            if (n < MAX_OUT) out[n++] = x;
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
            if (n < MAX_OUT) out[n++] = x;
        }
        if (v == R) ++bal;
        else if (v == L) --bal;
        if (bal < 0) break;
    }
    return n;
}

template<int MAX_OUT>
ONEESAN_CINV_HD int ordinary_closure_preimages_partial_reverse(
    MateID dest, int len, int p, MateID (&out)[MAX_OUT]
) {
    if (p <= 0 || p >= len) return 0;
    MateID mirrored = mirror_mate(dest, len);
    MateID tmp[MAX_OUT]{};
    int n = ordinary_closure_preimages_partial(mirrored, len, len - p, tmp);
    for (int i = 0; i < n; ++i) out[i] = mirror_mate(tmp[i], len);
    return n;
}

} // namespace oneesan::gridfp

#undef ONEESAN_CINV_HD
