#pragma once

#include "../../common/gridfp_closure_inverse.hpp"
#include "../../common/gridfp_transition.hpp"
#include "../../common/gridfp_transition_reverse.hpp"

#include <cstdint>

namespace oneesan::gridfp::reducedprod {

using Rank64 = std::uint64_t;

static constexpr int RP_MAX_W = 28;
static constexpr int RP_MAX_SECTORS = 16;
static constexpr int RP_MAX_TERMS = 32;

struct DeviceKey {
    MateID mate = 0;
    std::uint8_t blocked = 0;
};

struct DeviceTerm {
    DeviceKey key{};
    std::int8_t coef = 0;
};

__constant__ Rank64 RP_CHOOSE[RP_MAX_W + 1][RP_MAX_W + 1];
__constant__ Rank64 RP_PRIMITIVE[RP_MAX_W + 1][RP_MAX_W + 2];
__constant__ Rank64 RP_MOTZKIN[RP_MAX_W + 1][RP_MAX_W + 2];
__constant__ Rank64 RP_SECTOR_OFFSET[RP_MAX_SECTORS + 1];
__constant__ Rank64 RP_SECTOR_MAIN[RP_MAX_SECTORS];
__constant__ Rank64 RP_SECTOR_PRIMITIVE[RP_MAX_SECTORS];

__device__ __forceinline__ bool key_equal(DeviceKey a, DeviceKey b) {
    return a.mate == b.mate && a.blocked == b.blocked;
}

__device__ __forceinline__ bool valid_mate_device(MateID m, int len) {
    int h = 1;
    for (int pos = 0; pos < len; ++pos) {
        const MateValue v = mget(m, len - 1 - pos);
        if (v == X) return false;
        if (v == R) --h;
        else if (v == L) ++h;
        if (h < 0) return false;
    }
    return h == 0;
}

__device__ __forceinline__ int add_term(DeviceTerm* out, int n, DeviceKey key, int coef) {
    if (!coef) return n;
    for (int i = 0; i < n; ++i) {
        if (!key_equal(out[i].key, key)) continue;
        const int z = int(out[i].coef) + coef;
        out[i].coef = static_cast<std::int8_t>(z);
        if (!z) {
            out[i] = out[n - 1];
            --n;
        }
        return n;
    }
    if (n >= RP_MAX_TERMS) return -1;
    out[n].key = key;
    out[n].coef = static_cast<std::int8_t>(coef);
    return n + 1;
}

__device__ __forceinline__ int add_mate_unique(MateID* out, int n, MateID m) {
    for (int i = 0; i < n; ++i) if (out[i] == m) return n;
    if (n >= RP_MAX_TERMS) return -1;
    out[n] = m;
    return n + 1;
}

__device__ __forceinline__ MateID motzkin_unrank_device(int len, Rank64 rank) {
    MateID m = 0;
    int h = 1;
    for (int pos = 0; pos < len; ++pos) {
        const int rem = len - pos - 1;
        const Rank64 n_count = RP_MOTZKIN[rem][h];
        MateValue v = N;
        if (rank < n_count) {
            v = N;
        } else {
            rank -= n_count;
            const Rank64 r_count = h > 0 ? RP_MOTZKIN[rem][h - 1] : 0;
            if (rank < r_count) {
                v = R;
                --h;
            } else {
                rank -= r_count;
                v = L;
                ++h;
            }
        }
        const int bit = len - 1 - pos;
        m |= MateID(v) << (2 * bit);
    }
    return m;
}

__device__ __forceinline__ Rank64 primitive_rank_device(MateID m, int len, int occupied) {
    int h = 1;
    int seen = 0;
    Rank64 rank = 0;
    for (int pos = 0; pos < len; ++pos) {
        const MateValue c = mget(m, len - 1 - pos);
        if (c == N) continue;
        const int rem = occupied - (++seen);
        if (c == L) {
            if (h > 0) rank += RP_PRIMITIVE[rem][h - 1];
            ++h;
        } else {
            --h;
        }
    }
    return rank;
}

__device__ __forceinline__ Rank64 support_rank_main_device(MateID m, int len, int occupied) {
    Rank64 rank = 0;
    int left = occupied;
    for (int pos = 0; pos < len; ++pos) {
        if (mget(m, len - 1 - pos) == N) continue;
        const int rem = len - pos - 1;
        rank += RP_CHOOSE[rem][left];
        --left;
    }
    return rank;
}

__device__ __forceinline__ Rank64 support_rank_block_device(
    MateID m, int len, int occupied, int fixed_bit
) {
    const int fixed_pos = len - 1 - fixed_bit;
    Rank64 rank = 0;
    int left = occupied - 1;
    int compact_pos = 0;
    const int compact_len = len - 1;
    for (int pos = 0; pos < len; ++pos) {
        if (pos == fixed_pos) continue;
        if (mget(m, len - 1 - pos) != N) {
            const int rem = compact_len - compact_pos - 1;
            rank += RP_CHOOSE[rem][left];
            --left;
        }
        ++compact_pos;
    }
    return rank;
}

__device__ __forceinline__ Rank64 factor_rank_device(DeviceKey k, int W, int fixed_bit) {
    const int len = k.blocked ? W - 1 : W;
    int occupied = 0;
    for (int bit = 0; bit < len; ++bit) occupied += mget(k.mate, bit) != N;
    const int sector = (occupied - 1) >> 1;
    const Rank64 pc = RP_SECTOR_PRIMITIVE[sector];
    const Rank64 pr = primitive_rank_device(k.mate, len, occupied);
    const Rank64 base = RP_SECTOR_OFFSET[sector];
    if (!k.blocked) {
        return base + support_rank_main_device(k.mate, len, occupied) * pc + pr;
    }
    return base + RP_SECTOR_MAIN[sector] +
           support_rank_block_device(k.mate, len, occupied, fixed_bit) * pc + pr;
}

__device__ __forceinline__ int project_forward(DeviceKey k, int W, int q, int coef, DeviceTerm* out, int n) {
    if (!k.blocked || mget(k.mate, q - 1) != N) return add_term(out, n, k, coef);
    const MateID nn = blocked_exclude(k.mate, q);
    n = add_term(out, n, DeviceKey{nn, 0}, coef);
    if (n < 0) return n;
    const MateID lr = msetpair(nn, q, LR);
    return add_term(out, n, DeviceKey{lr, 0}, -coef);
}

__device__ __forceinline__ int reduced_step_forward(DeviceKey src, int W, int p, DeviceTerm* out) {
    int n = 0;
    if (!src.blocked) {
        n = add_term(out, n, src, 1);
        if (n < 0) return n;
        const IncludeResult z = include_horizontal(src.mate, W, p);
        if (!z.valid) return n;
        return project_forward(DeviceKey{z.mate, std::uint8_t(z.blocked)}, W, p - 1, 1, out, n);
    }
    return add_term(out, n, DeviceKey{blocked_exclude(src.mate, p), 0}, 1);
}

__device__ __forceinline__ int blocked_include_preimages_forward_device(
    MateID b, int W, int p, MateID* out
) {
    int n = 0;
    if (is_endpoint(mget(b, p - 1))) {
        const MateID x = minsert(b, p, N);
        const IncludeResult z = include_horizontal(x, W, p);
        if (valid_mate_device(x, W) && z.valid && z.blocked && z.mate == b) {
            n = add_mate_unique(out, n, x);
            if (n < 0) return n;
        }
    }

    const MateID closure_dest = minsert(b, p - 1, N);
    MateID cand[RP_MAX_TERMS]{};
    const int nc = ordinary_closure_preimages_partial(closure_dest, W, p, cand);
    for (int i = 0; i < nc; ++i) {
        const MateID x = cand[i];
        const IncludeResult z = include_horizontal(x, W, p);
        if (!valid_mate_device(x, W) || !z.valid || !z.blocked || z.mate != b) continue;
        n = add_mate_unique(out, n, x);
        if (n < 0) return n;
    }
    return n;
}

__device__ __forceinline__ int try_main_inverse_device(
    MateID x, MateID dest, int W, int p, DeviceTerm* out, int n
) {
    if (!valid_mate_device(x, W)) return n;
    const IncludeResult z = include_horizontal(x, W, p);
    if (z.valid && !z.blocked && z.mate == dest)
        return add_term(out, n, DeviceKey{x, 0}, 1);
    return n;
}

__device__ __forceinline__ int inverse_reduced_forward(DeviceKey dest, int W, int p, DeviceTerm* out) {
    int n = 0;
    if (dest.blocked) {
        MateID pre[RP_MAX_TERMS]{};
        const int np = blocked_include_preimages_forward_device(dest.mate, W, p, pre);
        if (np < 0) return np;
        for (int i = 0; i < np; ++i) {
            n = add_term(out, n, DeviceKey{pre[i], 0}, 1);
            if (n < 0) return n;
        }
        return n;
    }

    const MateID d = dest.mate;
    n = add_term(out, n, DeviceKey{d, 0}, 1);
    if (n < 0) return n;

    const MateValuePair w = mpair(d, p);
    if (w == LR) n = try_main_inverse_device(msetpair(d, p, NN), d, W, p, out, n);
    if (n < 0) return n;
    if (w == NR) n = try_main_inverse_device(msetpair(d, p, RN), d, W, p, out, n);
    if (n < 0) return n;
    if (w == NL) n = try_main_inverse_device(msetpair(d, p, LN), d, W, p, out, n);
    if (n < 0) return n;

    if (mget(d, p) == N && is_endpoint(mget(d, p - 1))) {
        const MateID b = mshrink(d, p);
        if (valid_mate_device(b, W - 1) && mget(b, p - 1) != N && blocked_exclude(b, p) == d) {
            n = add_term(out, n, DeviceKey{b, 1}, 1);
            if (n < 0) return n;
        }
    }

    const int q = p - 1;
    const MateValuePair qp = mpair(d, q);
    if (qp == NN || qp == LR) {
        const MateID nn = qp == NN ? d : msetpair(d, q, NN);
        const MateID b = mshrink(nn, q);
        if (valid_mate_device(b, W - 1) && mget(b, q - 1) == N) {
            MateID pre[RP_MAX_TERMS]{};
            const int np = blocked_include_preimages_forward_device(b, W, p, pre);
            if (np < 0) return np;
            const int sign = qp == NN ? 1 : -1;
            for (int i = 0; i < np; ++i) {
                n = add_term(out, n, DeviceKey{pre[i], 0}, sign);
                if (n < 0) return n;
            }
        }
    }
    return n;
}

__device__ __forceinline__ DeviceKey forward_component_seed(MateID label, int W, int p, bool& eligible) {
    eligible = !(mget(label, p - 1) == N && mget(label, p - 2) == N);
    if (!eligible) return {};
    if (mget(label, p - 1) != N) return DeviceKey{label, 1};
    return DeviceKey{blocked_exclude(label, p), 0};
}

} // namespace oneesan::gridfp::reducedprod
