#pragma once

#include "gridfp_transition.hpp"

#ifndef ONEESAN_FAST_INCLUDE_HORIZONTAL_REVERSE
#define ONEESAN_FAST_INCLUDE_HORIZONTAL_REVERSE 1
#endif
#ifndef ONEESAN_FAST_BLOCKED_EXCLUDE_REVERSE
#define ONEESAN_FAST_BLOCKED_EXCLUDE_REVERSE 1
#endif
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

// Horizontal reflection of a frontier word. Reversing the position order and
// swapping L/R preserves the noncrossing connectivity represented by the
// Motzkin/parenthesis code.
ONEESAN_REV_HD MateID mirror_mate(MateID m, int width) {
    MateID out = 0;
    for (int i = 0; i < width; ++i) {
        out |= MateID(mirror_value(mget(m, i))) << (2 * (width - 1 - i));
    }
    return out;
}

// Direct-coordinate reverse transition. This is exactly
// mirror(include_horizontal(mirror(m), width-p)); expanding the conjugation
// keeps the local pair rewrites and closure scans but removes both full-width
// mirror passes. Reverse boundary compression removes original position p.
ONEESAN_REV_HD IncludeResult include_horizontal_reverse(MateID m, int width, int p) {
#if ONEESAN_FAST_INCLUDE_HORIZONTAL_REVERSE
    IncludeResult z{};
    MateID t = m;
    switch (mpair(m, p)) {
    case NN:
        z.mate = msetpair(m, p, LR);
        z.valid = true;
        return z;
    case LN: case RN:
        if (p == width - 1) {
            z.mate = msetpair(m, p, mpair(m, p) == LN ? NL : NR);
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
