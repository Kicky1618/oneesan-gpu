#pragma once

#include "two_cell_recoupling_rank.hpp"

#include <cstdint>

#if defined(__CUDACC__)
#define ONEESAN_TC_COMP_HD __host__ __device__ __forceinline__
#else
#define ONEESAN_TC_COMP_HD inline
#endif

namespace oneesan::twocell {

enum Symbol : std::uint8_t { TC_N = 0, TC_R = 1, TC_L = 2 };

struct PackedWord {
    std::uint32_t support = 0;
    std::uint32_t left = 0;
    std::uint8_t len = 0;
};

template <class T, int Capacity>
struct SmallList {
    T value[Capacity]{};
    int size = 0;
    bool overflow = false;

    ONEESAN_TC_COMP_HD bool contains(const T& x) const {
        for (int i = 0; i < size; ++i)
            if (equal(value[i], x)) return true;
        return false;
    }

    ONEESAN_TC_COMP_HD bool insert(const T& x) {
        if (contains(x)) return true;
        if (size >= Capacity) {
            overflow = true;
            return false;
        }
        value[size++] = x;
        return true;
    }
};

ONEESAN_TC_COMP_HD bool equal(const PackedWord& a, const PackedWord& b) {
    return a.support == b.support && a.left == b.left && a.len == b.len;
}

ONEESAN_TC_COMP_HD bool equal(const PackedKey& a, const PackedKey& b) {
    return a.support == b.support && a.left == b.left && a.type == b.type;
}

ONEESAN_TC_COMP_HD PackedWord state_word(PackedKey k, int W) {
    return PackedWord{k.support, k.left, static_cast<std::uint8_t>(k.type ? W - 2 : W - 1)};
}

ONEESAN_TC_COMP_HD PackedKey make_state(std::uint8_t type, PackedWord w) {
    return PackedKey{w.support, w.left, type};
}

ONEESAN_TC_COMP_HD Symbol symbol(PackedWord w, int pos) {
    const std::uint32_t bit = std::uint32_t(1) << pos;
    if (!(w.support & bit)) return TC_N;
    return (w.left & bit) ? TC_L : TC_R;
}

ONEESAN_TC_COMP_HD PackedWord set_symbol(PackedWord w, int pos, Symbol c) {
    const std::uint32_t bit = std::uint32_t(1) << pos;
    w.support &= ~bit;
    w.left &= ~bit;
    if (c != TC_N) {
        w.support |= bit;
        if (c == TC_L) w.left |= bit;
    }
    return w;
}

ONEESAN_TC_COMP_HD PackedWord insert_symbol(PackedWord w, int pos, Symbol c) {
    const std::uint32_t lo = low_mask(pos);
    const std::uint32_t lo_support = w.support & lo;
    const std::uint32_t hi_support = w.support & ~lo;
    const std::uint32_t lo_left = w.left & lo;
    const std::uint32_t hi_left = w.left & ~lo;
    w.support = lo_support | (std::uint32_t(c != TC_N) << pos) | (hi_support << 1);
    w.left = lo_left | (std::uint32_t(c == TC_L) << pos) | (hi_left << 1);
    ++w.len;
    return w;
}

ONEESAN_TC_COMP_HD PackedWord remove_symbol(PackedWord w, int pos) {
    const std::uint32_t lo = low_mask(pos);
    w.support = (w.support & lo) | ((w.support >> (pos + 1)) << pos);
    w.left = (w.left & lo) | ((w.left >> (pos + 1)) << pos);
    --w.len;
    return w;
}

ONEESAN_TC_COMP_HD bool valid_word(PackedWord w) {
    int h = 1;
    for (int pos = 0; pos < w.len; ++pos) {
        const Symbol c = symbol(w, pos);
        if (c == TC_L) ++h;
        else if (c == TC_R) --h;
        if (h < 0) return false;
    }
    return h == 0;
}

ONEESAN_TC_COMP_HD int partner(PackedWord w, int pos) {
    const Symbol c = symbol(w, pos);
    if (c == TC_N) return -2;
    if (c == TC_L) {
        int depth = 1;
        for (int q = pos + 1; q < w.len; ++q) {
            const Symbol z = symbol(w, q);
            if (z == TC_L) ++depth;
            else if (z == TC_R && --depth == 0) return q;
        }
        return -3;
    }
    int depth = 1;
    for (int q = pos - 1; q >= 0; --q) {
        const Symbol z = symbol(w, q);
        if (z == TC_R) ++depth;
        else if (z == TC_L && --depth == 0) return q;
    }
    return -1; // distinguished root
}

ONEESAN_TC_COMP_HD int root_position(PackedWord w) {
    int h = 1;
    for (int pos = 0; pos < w.len; ++pos) {
        const Symbol c = symbol(w, pos);
        if (c == TC_L) ++h;
        else if (c == TC_R && --h == 0) return pos;
    }
    return -1;
}

ONEESAN_TC_COMP_HD SmallList<PackedWord, 2> apply_T(PackedWord w, int i) {
    SmallList<PackedWord, 2> out;
    const bool a = symbol(w, i) != TC_N;
    const bool b = symbol(w, i + 1) != TC_N;
    if (!a && !b) {
        out.insert(w);
        out.insert(set_symbol(set_symbol(w, i, TC_L), i + 1, TC_R));
        return out;
    }
    if (a && !b) {
        out.insert(w);
        const Symbol c = symbol(w, i);
        out.insert(set_symbol(set_symbol(w, i, TC_N), i + 1, c));
        return out;
    }
    if (!a && b) {
        const Symbol c = symbol(w, i + 1);
        out.insert(set_symbol(set_symbol(w, i + 1, TC_N), i, c));
        out.insert(w);
        return out;
    }

    const int p = partner(w, i);
    const int q = partner(w, i + 1);
    if (p == i + 1 && q == i) return out; // beta=0 loop
    if (p < -1 || q < -1) return out;

    PackedWord z = set_symbol(set_symbol(w, i, TC_N), i + 1, TC_N);
    if (p < 0) {
        if (q < 0) return out;
        z = set_symbol(z, q, TC_R);
    } else if (q < 0) {
        z = set_symbol(z, p, TC_R);
    } else {
        const int lo = p < q ? p : q;
        const int hi = p < q ? q : p;
        z = set_symbol(z, lo, TC_L);
        z = set_symbol(z, hi, TC_R);
    }
    if (valid_word(z)) out.insert(z);
    return out;
}

ONEESAN_TC_COMP_HD PackedWord collapse_A(PackedWord w, int i) {
    const Symbol a = symbol(w, i);
    const Symbol b = symbol(w, i + 1);
    if (a == TC_N && b != TC_N) {
        w = set_symbol(w, i, b);
        w = set_symbol(w, i + 1, TC_N);
    }
    return remove_symbol(w, i + 1);
}

ONEESAN_TC_COMP_HD PackedWord remove_pair(PackedWord w, int i) {
    w = remove_symbol(w, i + 1);
    return remove_symbol(w, i);
}

ONEESAN_TC_COMP_HD SmallList<PackedKey, 2> R_raw(PackedWord w, int i) {
    SmallList<PackedKey, 2> out;
    const bool a = symbol(w, i) != TC_N;
    const bool b = symbol(w, i + 1) != TC_N;
    if (!a && !b) {
        out.insert(make_state(0, collapse_A(w, i)));
        out.insert(make_state(1, remove_pair(w, i)));
    } else if (a != b) {
        out.insert(make_state(0, collapse_A(w, i)));
    } else {
        const auto z = apply_T(w, i);
        if (z.size == 1) out.insert(make_state(0, collapse_A(z.value[0], i)));
    }
    return out;
}

ONEESAN_TC_COMP_HD SmallList<PackedWord, 2> E_raw(PackedKey k, int W, int i) {
    SmallList<PackedWord, 2> out;
    PackedWord w = state_word(k, W);
    if (k.type == 1) {
        w = insert_symbol(w, i, TC_L);
        w = insert_symbol(w, i + 1, TC_R);
        out.insert(w);
        return out;
    }
    const Symbol c = symbol(w, i);
    if (c == TC_N) {
        out.insert(insert_symbol(w, i + 1, TC_N));
    } else {
        out.insert(insert_symbol(w, i + 1, TC_N));
        out.insert(insert_symbol(w, i, TC_N));
    }
    return out;
}

ONEESAN_TC_COMP_HD PackedKey project(PackedKey k, int W, int i) {
    if (k.type == 0) return k;
    PackedWord w = state_word(k, W);
    if (i <= W - 3 && symbol(w, i) == TC_N) {
        w = remove_symbol(w, i);
        w = insert_symbol(w, i, TC_L);
        w = insert_symbol(w, i + 1, TC_R);
        return make_state(0, w);
    }
    return k;
}

ONEESAN_TC_COMP_HD SmallList<PackedKey, 3> K_step(PackedKey src, int W, int i) {
    SmallList<PackedKey, 3> out;
    const auto expanded = E_raw(src, W, i);
    for (int a = 0; a < expanded.size; ++a) {
        const auto raw = R_raw(expanded.value[a], i + 1);
        for (int b = 0; b < raw.size; ++b)
            out.insert(project(raw.value[b], W, i + 1));
    }
    return out;
}

ONEESAN_TC_COMP_HD SmallList<PackedWord, 32> inverse_R_raw(
    PackedKey raw, int W, int j
) {
    SmallList<PackedWord, 32> out;
    PackedWord rw = state_word(raw, W);
    if (raw.type == 1) {
        rw = insert_symbol(rw, j, TC_N);
        rw = insert_symbol(rw, j + 1, TC_N);
        if (valid_word(rw)) out.insert(rw);
        return out;
    }

    const Symbol local = symbol(rw, j);
    if (local != TC_N) {
        out.insert(insert_symbol(rw, j + 1, TC_N));
        out.insert(insert_symbol(rw, j, TC_N));
        return out;
    }

    PackedWord z = insert_symbol(rw, j + 1, TC_N);
    out.insert(z);
    int h[kMaxWidth + 2]{};
    h[0] = 1;
    for (int pos = 0; pos < W; ++pos) {
        h[pos + 1] = h[pos];
        const Symbol c = symbol(z, pos);
        if (c == TC_L) ++h[pos + 1];
        else if (c == TC_R) --h[pos + 1];
    }
    const int level = h[j];
    int face_left = j;
    while (face_left > 0 && h[face_left - 1] >= level) --face_left;
    int face_right = j + 2;
    while (face_right < W && h[face_right + 1] >= level) ++face_right;

    for (int p = 0; p < W; ++p) {
        if (symbol(z, p) != TC_L) continue;
        const int q = partner(z, p);
        if (q < 0 || p < face_left || q >= face_right || h[p] != level) continue;
        PackedWord w = z;
        bool candidate = true;
        if (q < j) {
            w = set_symbol(w, q, TC_L);
            w = set_symbol(w, j, TC_R);
            w = set_symbol(w, j + 1, TC_R);
        } else if (p > j + 1) {
            w = set_symbol(w, p, TC_R);
            w = set_symbol(w, j, TC_L);
            w = set_symbol(w, j + 1, TC_L);
        } else if (p < j && q > j + 1) {
            w = set_symbol(w, j, TC_R);
            w = set_symbol(w, j + 1, TC_L);
        } else {
            candidate = false;
        }
        if (candidate && valid_word(w)) out.insert(w);
    }

    if (face_left > 0) {
        const int p = face_left - 1;
        if (symbol(z, p) == TC_L) {
            const int q = partner(z, p);
            if (q == face_right && p < j && q > j + 1) {
                PackedWord w = set_symbol(set_symbol(z, j, TC_R), j + 1, TC_L);
                if (valid_word(w)) out.insert(w);
            }
        }
    }

    const int root = root_position(z);
    if (root >= 0 &&
        (level == 0 || (face_left == 0 && face_right < W && root == face_right))) {
        if (root < j) {
            PackedWord w = set_symbol(z, root, TC_L);
            w = set_symbol(w, j, TC_R);
            w = set_symbol(w, j + 1, TC_R);
            if (valid_word(w)) out.insert(w);
        } else if (root > j + 1) {
            PackedWord w = set_symbol(set_symbol(z, j, TC_R), j + 1, TC_L);
            if (valid_word(w)) out.insert(w);
        }
    }
    return out;
}

ONEESAN_TC_COMP_HD SmallList<PackedKey, 2> inverse_project(
    PackedKey dst, int W, int j
) {
    SmallList<PackedKey, 2> out;
    out.insert(dst);
    if (dst.type == 0) {
        PackedWord w = state_word(dst, W);
        if (j <= W - 3 && j + 1 < w.len &&
            symbol(w, j) == TC_L && symbol(w, j + 1) == TC_R) {
            w = remove_symbol(w, j + 1);
            w = set_symbol(w, j, TC_N);
            const PackedKey c = make_state(1, w);
            if (equal(project(c, W, j), dst)) out.insert(c);
        }
    }
    return out;
}

ONEESAN_TC_COMP_HD bool inverse_E(PackedWord z, int i, PackedKey& out) {
    const bool a = symbol(z, i) != TC_N;
    const bool b = symbol(z, i + 1) != TC_N;
    if (!a && !b) {
        out = make_state(0, remove_symbol(z, i + 1));
        return true;
    }
    if (a != b) {
        out = make_state(0, collapse_A(z, i));
        return true;
    }
    if (symbol(z, i) == TC_L && symbol(z, i + 1) == TC_R) {
        out = make_state(1, remove_pair(z, i));
        return true;
    }
    return false;
}

ONEESAN_TC_COMP_HD bool in_source_layout(PackedKey k, int W, int i) {
    const PackedWord w = state_word(k, W);
    if (!valid_word(w)) return false;
    if (k.type == 0) return true;
    return i > W - 3 || symbol(w, i) != TC_N;
}

ONEESAN_TC_COMP_HD SmallList<PackedKey, 32> inverse_K(
    PackedKey dst, int W, int i
) {
    SmallList<PackedKey, 32> out;
    const auto raw = inverse_project(dst, W, i + 1);
    for (int a = 0; a < raw.size; ++a) {
        const auto full = inverse_R_raw(raw.value[a], W, i + 1);
        for (int b = 0; b < full.size; ++b) {
            PackedKey src{};
            if (inverse_E(full.value[b], i, src) && in_source_layout(src, W, i))
                out.insert(src);
        }
    }
    return out;
}

ONEESAN_TC_COMP_HD bool deep_collapse(PackedWord label, int i, PackedWord& collapsed) {
    const Symbol a = symbol(label, i);
    const Symbol b = symbol(label, i + 1);
    if ((a == TC_R || a == TC_L) && b == TC_N) {
        collapsed = remove_symbol(label, i + 1);
        return true;
    }
    if (a == TC_L && b == TC_R) {
        collapsed = remove_symbol(label, i + 1);
        collapsed = set_symbol(collapsed, i, TC_N);
        return true;
    }
    return false;
}

// Exact source list of one interior K_i component. No graph traversal is used:
// singleton/triple cases are direct, while a deep component performs exactly
// one marked-face inverse scan. Exhaustive CPU probes give max size 17 at W=28.
ONEESAN_TC_COMP_HD SmallList<PackedKey, 18> direct_component_sources(
    PackedWord label, int W, int i
) {
    SmallList<PackedKey, 18> out;
    const PackedKey c = make_state(1, label);
    if (symbol(label, i) == TC_N) {
        out.insert(project(c, W, i));
        return out;
    }

    out.insert(c);
    out.insert(make_state(0, insert_symbol(label, i, TC_N)));
    out.insert(make_state(0, insert_symbol(label, i + 1, TC_N)));

    PackedWord collapsed{};
    if (!deep_collapse(label, i, collapsed)) return out;

    PackedWord central = insert_symbol(collapsed, i, TC_N);
    central = insert_symbol(central, i, TC_N);
    const auto pre = inverse_K(make_state(0, central), W, i);
    for (int q = 0; q < pre.size; ++q) out.insert(pre.value[q]);
    if (pre.overflow) out.overflow = true;
    return out;
}

} // namespace oneesan::twocell

#undef ONEESAN_TC_COMP_HD
