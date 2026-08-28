#pragma once

#include "gridfp_transition.hpp"

#ifndef ONEESAN_FAST_MIRROR_MATE
#define ONEESAN_FAST_MIRROR_MATE 1
#endif
#ifndef ONEESAN_FAST_INCLUDE_HORIZONTAL_REVERSE
#define ONEESAN_FAST_INCLUDE_HORIZONTAL_REVERSE 1
#endif
#ifndef ONEESAN_FAST_BLOCKED_EXCLUDE_REVERSE
#define ONEESAN_FAST_BLOCKED_EXCLUDE_REVERSE 1
#endif
static_assert(ONEESAN_FAST_MIRROR_MATE == 0 || ONEESAN_FAST_MIRROR_MATE == 1,
              "ONEESAN_FAST_MIRROR_MATE must be 0 or 1");
static_assert(
    ONEESAN_FAST_INCLUDE_HORIZONTAL_REVERSE == 0 ||
    ONEESAN_FAST_INCLUDE_HORIZONTAL_REVERSE == 1,
    "ONEESAN_FAST_INCLUDE_HORIZONTAL_REVERSE must be 0 or 1");
static_assert(
    ONEESAN_FAST_BLOCKED_EXCLUDE_REVERSE == 0 ||
    ONEESAN_FAST_BLOCKED_EXCLUDE_REVERSE == 1,
    "ONEESAN_FAST_BLOCKED_EXCLUDE_REVERSE must be 0 or 1");

namespace oneesan::gridfp {

#if defined(__CUDACC__)
#define ONEESAN_REV_HD __host__ __device__ __forceinline__
#else
#define ONEESAN_REV_HD inline
#endif

ONEESAN_REV_HD MateValue mirror_value(MateValue v) {
    if (v == R) return L;
    if (v == L) return R;
    return v;
}

ONEESAN_REV_HD MateID reverse_bits64(MateID x) {
#if defined(__CUDA_ARCH__)
    return __brevll(x);
#else
    x = ((x >> 1) & 0x5555555555555555ULL) |
        ((x & 0x5555555555555555ULL) << 1);
    x = ((x >> 2) & 0x3333333333333333ULL) |
        ((x & 0x3333333333333333ULL) << 2);
    x = ((x >> 4) & 0x0f0f0f0f0f0f0f0fULL) |
        ((x & 0x0f0f0f0f0f0f0f0fULL) << 4);
    x = ((x >> 8) & 0x00ff00ff00ff00ffULL) |
        ((x & 0x00ff00ff00ff00ffULL) << 8);
    x = ((x >> 16) & 0x0000ffff0000ffffULL) |
        ((x & 0x0000ffff0000ffffULL) << 16);
    return (x >> 32) | (x << 32);
#endif
}

// Reversing all 64 bits simultaneously reverses the order of 2-bit frontier
// symbols and reverses the two bits within each symbol. The latter is exactly
// R(01)<->L(10), while N(00) and X(11) remain unchanged.
ONEESAN_REV_HD MateID mirror_mate(MateID m, int width) {
#if ONEESAN_FAST_MIRROR_MATE
    if (width <= 0) return 0;
    const MateID rev = reverse_bits64(m);
    const int shift = 64 - 2 * width;
    return shift ? (rev >> shift) : rev;
#else
    MateID out = 0;
    for (int i = 0; i < width; ++i) {
        out |= MateID(mirror_value(mget(m, i))) << (2 * (width - 1 - i));
    }
    return out;
#endif
}

// Direct-coordinate reverse transition. This is exactly
// mirror(include_horizontal(mirror(m), width-p)); expanding the conjugation
// keeps the local pair rewrites and closure scans but removes both full-width
// mirror passes. Reverse boundary compression removes original position p.
ONEESAN_REV_HD IncludeResult include_horizontal_reverse(MateID m, int width, int p) {
#if ONEESAN_FAST_INCLUDE_HORIZONTAL_REVERSE
    IncludeResult z{};
    MateID t = m;
    const MateValuePair pair = mpair(m, p);
    switch (pair) {
    case NN:
        z.mate = msetpair(m, p, LR);
        z.valid = true;
        return z;
    case LN: case RN:
        if (p == width - 1) {
            z.mate = msetpair(m, p, pair == LN ? NL : NR);
            z.valid = true;
            return z;
        }
        z.mate = mshrink(m, p - 1);
        z.valid = true;
        z.blocked = true;
        return z;
    case NL:
        z.mate = msetpair(m, p, LN);
        z.valid = true;
        return z;
    case NR:
        z.mate = msetpair(m, p, RN);
        z.valid = true;
        return z;
    case LL: {
        t = msetpair(m, p, NN);
        int q = p - 1, s = 1;
        while (s) {
            --q;
            if (q < 0) return z;
            const MateValue v = mget(t, q);
            if (v == L) ++s;
            else if (v == R) --s;
        }
        t = mset(t, q, L);
        if (p == width - 1) {
            z.mate = t;
            z.valid = true;
            return z;
        }
        z.mate = mshrink(t, p);
        z.valid = true;
        z.blocked = true;
        return z;
    }
    case RR: {
        t = msetpair(m, p, NN);
        int q = p, s = 1;
        while (s) {
            ++q;
            if (q >= width) return z;
            const MateValue v = mget(t, q);
            if (v == L) --s;
            else if (v == R) ++s;
        }
        t = mset(t, q, R);
        if (p == width - 1) {
            z.mate = t;
            z.valid = true;
            return z;
        }
        z.mate = mshrink(t, p);
        z.valid = true;
        z.blocked = true;
        return z;
    }
    case RL:
        t = msetpair(m, p, NN);
        if (p == width - 1) {
            z.mate = t;
            z.valid = true;
            return z;
        }
        z.mate = mshrink(t, p);
        z.valid = true;
        z.blocked = true;
        return z;
    default:
        return z;
    }
#else
    IncludeResult z = include_horizontal(mirror_mate(m, width), width, width - p);
    if (!z.valid) return z;
    z.mate = mirror_mate(z.mate, z.blocked ? width - 1 : width);
    return z;
#endif
}

// Reverse-scan counterpart of blocked_exclude(). Reflection maps the inserted
// position width-p back to p-1, while the two L/R swaps cancel. Therefore the
// full mirror -> insert -> mirror sequence is exactly one direct insertion.
ONEESAN_REV_HD MateID blocked_exclude_reverse(MateID compressed, int width, int p) {
#if ONEESAN_FAST_BLOCKED_EXCLUDE_REVERSE
    (void)width;
    return minsert(compressed, p - 1, N);
#else
    MateID mirrored = mirror_mate(compressed, width - 1);
    MateID expanded = blocked_exclude(mirrored, width - p);
    return mirror_mate(expanded, width);
#endif
}

#undef ONEESAN_REV_HD

} // namespace oneesan::gridfp
