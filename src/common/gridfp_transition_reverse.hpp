#pragma once

#include "gridfp_transition.hpp"

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

// Transition for scanning the same row from the opposite horizontal direction.
// Pair p in the original coordinates reflects to pair width-p. The reflected
// blocked representation has width-1, so a blocked include result is mirrored
// with that compressed width before returning to the original coordinates.
ONEESAN_REV_HD IncludeResult include_horizontal_reverse(MateID m, int width, int p) {
    IncludeResult z = include_horizontal(mirror_mate(m, width), width, width - p);
    if (!z.valid) return z;
    z.mate = mirror_mate(z.mate, z.blocked ? width - 1 : width);
    return z;
}

// Reverse-scan counterpart of blocked_exclude().
ONEESAN_REV_HD MateID blocked_exclude_reverse(MateID compressed, int width, int p) {
    MateID mirrored = mirror_mate(compressed, width - 1);
    MateID expanded = blocked_exclude(mirrored, width - p);
    return mirror_mate(expanded, width);
}

#undef ONEESAN_REV_HD

} // namespace oneesan::gridfp
