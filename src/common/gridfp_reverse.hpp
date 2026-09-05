#pragma once
#include "gridfp_transition.hpp"

#ifdef __CUDACC__
#define ONEESAN_REVERSE_HD __host__ __device__ __forceinline__
#else
#define ONEESAN_REVERSE_HD inline
#endif

namespace oneesan::gridfp {
ONEESAN_REVERSE_HD int reverse_popcount(MateID x) {
#ifdef __CUDA_ARCH__
    return __popcll(x);
#else
    return __builtin_popcountll(x);
#endif
}
ONEESAN_REVERSE_HD int reverse_high_bit(MateID x) {
#ifdef __CUDA_ARCH__
    return 63 - __clzll(x);
#else
    return 63 - __builtin_clzll(x);
#endif
}
ONEESAN_REVERSE_HD int reverse_low_bit(MateID x) {
#ifdef __CUDA_ARCH__
    return __ffsll(x) - 1;
#else
    return __builtin_ctzll(x);
#endif
}

// Enumerate every main predecessor of a legal blocked target, for
// 2 <= p < width <= 28. Empty symbols never change closure depth or emit
// predecessors, so Sparse=true visits only occupied positions. Enumeration
// order and multiplicity match the scalar scan. No per-state tables needed.
// The low==N closure branch also supports p=1 as used by the boundary
// helper below; the endpoint branch is only an interior blocked transition.
template<bool Sparse = false, class Emit>
ONEESAN_REVERSE_HD void reverse_block_predecessors(MateID b, int width, int p, Emit emit) {
    auto low = mget(b, p - 1);
    if (low == R || low == L) {
        emit(minsert(b, p, N));
        return;
    }
    if (low != N) return;
    MateID u = minsert(b, p - 1, N);
    constexpr MateID even = 0x5555555555555555ULL;
    MateID above = even & ((MateID(1) << (2 * width)) - 1)
                       & ~((MateID(1) << (2 * (p + 1))) - 1);
    int height = 1 + reverse_popcount((u >> 1) & above) - reverse_popcount(u & above);
    if (height > 0) emit(msetpair(u, p, RL));

    MateID plugs = (u | (u >> 1)) & even;
    MateID bits = plugs & ((MateID(1) << (2 * (p - 1))) - 1);
    int depth = 1;
    for (int q = p - 2; q >= 0 && depth > 0; --q) {
        if constexpr (Sparse) {
            if (!bits) break;
            int bit = reverse_high_bit(bits);
            bits &= (MateID(1) << bit) - 1;
            q = bit / 2;
        }
        auto v = mget(u, q);
        if (v == L && depth == 1) emit(mset(msetpair(u, p, LL), q, R));
        if (v == L) ++depth;
        else if (v == R) --depth;
    }
    bits = plugs & above;
    depth = 1;
    for (int q = p + 1; q < width && depth > 0; ++q) {
        if constexpr (Sparse) {
            if (!bits) break;
            int bit = reverse_low_bit(bits);
            bits &= bits - 1;
            q = bit / 2;
        }
        auto v = mget(u, q);
        if (v == R && depth == 1) emit(mset(msetpair(u, p, RR), q, L));
        if (v == L) --depth;
        else if (v == R) ++depth;
    }
}
// Included-branch predecessors at the left boundary. No blocked outputs exist.
// For NN, the interior closure inverse remains valid with an empty slot at 0:
// there is no lower LL partner, and the RL/RR legality checks are unchanged.
template<class Emit>
ONEESAN_REVERSE_HD void reverse_boundary_main_predecessors(MateID t,int width,Emit emit){
    switch(mpair(t,1)){
    case LR:emit(msetpair(t,1,NN));break;
    case NR:emit(msetpair(t,1,RN));break;
    case NL:emit(msetpair(t,1,LN));break;
    case RN:emit(msetpair(t,1,NR));break;
    case LN:emit(msetpair(t,1,NL));break;
    case NN:reverse_block_predecessors<false>(mshrink(t,0),width,1,emit);break;
    default:break;
    }
}
} // namespace oneesan::gridfp

#undef ONEESAN_REVERSE_HD
