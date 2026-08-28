#pragma push_macro("main")
#undef main
#define main two_cell_inverse_gather_probe_main_unused
#include "two_cell_inverse_gather_probe.cpp"
#pragma pop_macro("main")

#include <array>

namespace {

struct PackedWord {
    std::uint32_t support = 0; // occupied positions
    std::uint32_t left = 0;    // occupied L positions; occupied non-L is R
    std::uint8_t len = 0;

    bool operator==(const PackedWord& o) const {
        return support == o.support && left == o.left && len == o.len;
    }
    bool operator<(const PackedWord& o) const {
        if (len != o.len) return len < o.len;
        if (support != o.support) return support < o.support;
        return left < o.left;
    }
};

struct PackedKey {
    char type = 'A';
    PackedWord w;

    bool operator==(const PackedKey& o) const { return type == o.type && w == o.w; }
    bool operator<(const PackedKey& o) const {
        return type < o.type || (type == o.type && w < o.w);
    }
};

template<class T, int CAP>
struct SmallUnique {
    std::array<T, CAP> value{};
    int size = 0;

    bool contains(const T& x) const {
        for (int i = 0; i < size; ++i)
            if (value[i] == x) return true;
        return false;
    }
    bool insert(const T& x) {
        if (contains(x)) return false;
        if (size >= CAP) fail("packed small-list capacity");
        value[size++] = x;
        return true;
    }
};

std::uint32_t low_mask(int pos) {
    if (pos <= 0) return 0;
    if (pos >= 32) return ~std::uint32_t(0);
    return (std::uint32_t(1) << pos) - 1;
}

char packed_symbol(const PackedWord& w, int pos) {
    assert(pos >= 0 && pos < w.len);
    const std::uint32_t bit = std::uint32_t(1) << pos;
    if (!(w.support & bit)) return N;
    return (w.left & bit) ? L : R;
}

PackedWord packed_set(PackedWord w, int pos, char c) {
    assert(pos >= 0 && pos < w.len);
    const std::uint32_t bit = std::uint32_t(1) << pos;
    w.support &= ~bit;
    w.left &= ~bit;
    if (c != N) {
        w.support |= bit;
        if (c == L) w.left |= bit;
    }
    return w;
}

PackedWord packed_insert(PackedWord w, int pos, char c) {
    assert(w.len < 31 && pos >= 0 && pos <= w.len);
    const std::uint32_t lo = low_mask(pos);
    auto insert_bit = [&](std::uint32_t x, bool b) {
        return (x & lo) | (std::uint32_t(b) << pos) | ((x & ~lo) << 1);
    };
    w.support = insert_bit(w.support, c != N);
    w.left = insert_bit(w.left, c == L);
    ++w.len;
    return w;
}

PackedWord packed_remove(PackedWord w, int pos) {
    assert(pos >= 0 && pos < w.len);
    const std::uint32_t lo = low_mask(pos);
    auto remove_bit = [&](std::uint32_t x) {
        return (x & lo) | ((x >> (pos + 1)) << pos);
    };
    w.support = remove_bit(w.support);
    w.left = remove_bit(w.left);
    --w.len;
    return w;
}

PackedWord pack_word(const Word& w) {
    if (w.size() >= 32) fail("packed word width");
    PackedWord out;
    out.len = static_cast<std::uint8_t>(w.size());
    for (int pos = 0; pos < static_cast<int>(w.size()); ++pos) {
        if (w[pos] == N) continue;
        out.support |= std::uint32_t(1) << pos;
        if (w[pos] == L) out.left |= std::uint32_t(1) << pos;
    }
    return out;
}

Word unpack_word(const PackedWord& w) {
    Word out(static_cast<std::size_t>(w.len), N);
    for (int pos = 0; pos < w.len; ++pos) out[pos] = packed_symbol(w, pos);
    return out;
}

PackedKey pack_key(const Key& k) { return PackedKey{k.type, pack_word(k.w)}; }
Key unpack_key(const PackedKey& k) { return Key{k.type, unpack_word(k.w)}; }

bool packed_valid(const PackedWord& w) {
    int h = 1;
    for (int pos = 0; pos < w.len; ++pos) {
        const char c = packed_symbol(w, pos);
        if (c == L) ++h;
        else if (c == R) --h;
        if (h < 0) return false;
    }
    return h == 0;
}

int packed_partner(const PackedWord& w, int pos) {
    const char c = packed_symbol(w, pos);
    assert(c != N);
    if (c == L) {
        int depth = 1;
        for (int q = pos + 1; q < w.len; ++q) {
            const char z = packed_symbol(w, q);
            if (z == L) ++depth;
            else if (z == R && --depth == 0) return q;
        }
        fail("packed L without partner");
    }

    int depth = 1;
    for (int q = pos - 1; q >= 0; --q) {
        const char z = packed_symbol(w, q);
        if (z == R) ++depth;
        else if (z == L && --depth == 0) return q;
    }
    return -1; // distinguished root
}

int packed_root(const PackedWord& w) {
    int h = 1;
    int root = -1;
    for (int pos = 0; pos < w.len; ++pos) {
        const char c = packed_symbol(w, pos);
        if (c == L) ++h;
        else if (c == R) {
            --h;
            if (h == 0 && root < 0) root = pos;
        }
    }
    if (h != 0 || root < 0) fail("packed root");
    return root;
}

SmallUnique<PackedWord, 2> packed_apply_T(const PackedWord& w, int i) {
    SmallUnique<PackedWord, 2> out;
    const bool a = packed_symbol(w, i) != N;
    const bool b = packed_symbol(w, i + 1) != N;
    if (!a && !b) {
        out.insert(w);
        out.insert(packed_set(packed_set(w, i, L), i + 1, R));
        return out;
    }
    if (a && !b) {
        out.insert(w);
        const char c = packed_symbol(w, i);
        out.insert(packed_set(packed_set(w, i, N), i + 1, c));
        return out;
    }
    if (!a && b) {
        const char c = packed_symbol(w, i + 1);
        out.insert(packed_set(packed_set(w, i + 1, N), i, c));
        out.insert(w);
        return out;
    }

    const int p = packed_partner(w, i);
    const int q = packed_partner(w, i + 1);
    if (p == i + 1 && q == i) return out; // beta=0 loop

    PackedWord z = packed_set(packed_set(w, i, N), i + 1, N);
    if (p < 0) {
        z = packed_set(z, q, R);
    } else if (q < 0) {
        z = packed_set(z, p, R);
    } else {
        z = packed_set(z, std::min(p, q), L);
        z = packed_set(z, std::max(p, q), R);
    }
    if (!packed_valid(z)) fail("packed cap invalid");
    out.insert(z);
    return out;
}

PackedWord packed_collapse_A(PackedWord w, int i) {
    const char a = packed_symbol(w, i);
    const char b = packed_symbol(w, i + 1);
    if (a != N && b != N) fail("packed collapse occupied pair");
    const char c = a != N ? a : b;
    if (a == N && b != N) {
        w = packed_set(w, i, b);
        w = packed_set(w, i + 1, N);
    }
    w = packed_remove(w, i + 1);
    if (packed_symbol(w, i) != c) fail("packed collapse symbol");
    return w;
}

PackedWord packed_remove_pair(PackedWord w, int i) {
    w = packed_remove(w, i + 1);
    return packed_remove(w, i);
}

SmallUnique<PackedKey, 2> packed_R_raw(const PackedWord& w, int i) {
    SmallUnique<PackedKey, 2> out;
    const bool a = packed_symbol(w, i) != N;
    const bool b = packed_symbol(w, i + 1) != N;
    if (!a && !b) {
        out.insert(PackedKey{'A', packed_collapse_A(w, i)});
        out.insert(PackedKey{'C', packed_remove_pair(w, i)});
    } else if (a != b) {
        out.insert(PackedKey{'A', packed_collapse_A(w, i)});
    } else {
        const auto z = packed_apply_T(w, i);
        if (z.size) {
            if (z.size != 1) fail("packed cap image size");
            out.insert(PackedKey{'A', packed_collapse_A(z.value[0], i)});
        }
    }
    return out;
}

SmallUnique<PackedWord, 2> packed_E_raw(const PackedKey& k, int i) {
    SmallUnique<PackedWord, 2> out;
    if (k.type == 'C') {
        PackedWord z = packed_insert(k.w, i, L);
        z = packed_insert(z, i + 1, R);
        out.insert(z);
        return out;
    }
    if (k.type != 'A') fail("packed E key type");
    const char c = packed_symbol(k.w, i);
    if (c == N) {
        out.insert(packed_insert(k.w, i + 1, N));
    } else {
        out.insert(packed_insert(k.w, i + 1, N));
        out.insert(packed_insert(k.w, i, N));
    }
    return out;
}

PackedKey packed_project(PackedKey k, int i, int W) {
    if (k.type == 'A') return k;
    if (k.type != 'C') fail("packed project type");
    if (i <= W - 3 && packed_symbol(k.w, i) == N) {
        PackedWord z = packed_remove(k.w, i);
        z = packed_insert(z, i, L);
        z = packed_insert(z, i + 1, R);
        return PackedKey{'A', z};
    }
    return k;
}

SmallUnique<PackedKey, 3> packed_K(const PackedKey& src, int W, int i) {
    SmallUnique<PackedKey, 3> out;
    const auto expanded = packed_E_raw(src, i);
    for (int a = 0; a < expanded.size; ++a) {
        const auto raw = packed_R_raw(expanded.value[a], i + 1);
        for (int b = 0; b < raw.size; ++b)
            out.insert(packed_project(raw.value[b], i + 1, W));
    }
    return out;
}

SmallUnique<PackedWord, 64> packed_inverse_R_raw(const PackedKey& raw, int j, int W) {
    SmallUnique<PackedWord, 64> out;
    if (raw.type == 'C') {
        PackedWord z = packed_insert(raw.w, j, N);
        z = packed_insert(z, j + 1, N);
        if (!packed_valid(z)) fail("packed inverse C invalid");
        out.insert(z);
        return out;
    }
    if (raw.type != 'A') fail("packed inverse R type");

    const char local = packed_symbol(raw.w, j);
    if (local != N) {
        out.insert(packed_insert(raw.w, j + 1, N));
        out.insert(packed_insert(raw.w, j, N));
        return out;
    }

    PackedWord z = packed_insert(raw.w, j + 1, N);
    out.insert(z);
    std::array<int, 33> h{};
    h[0] = 1;
    for (int pos = 0; pos < W; ++pos) {
        h[pos + 1] = h[pos];
        const char c = packed_symbol(z, pos);
        if (c == L) ++h[pos + 1];
        else if (c == R) --h[pos + 1];
    }
    const int level = h[j];
    int left = j;
    while (left > 0 && h[left - 1] >= level) --left;
    int right = j + 2;
    while (right < W && h[right + 1] >= level) ++right;

    for (int p = 0; p < W; ++p) {
        if (packed_symbol(z, p) != L) continue;
        const int q = packed_partner(z, p);
        if (p < left || q >= right || h[p] != level) continue;
        PackedWord w = z;
        bool candidate = true;
        if (q < j) {
            w = packed_set(w, q, L);
            w = packed_set(w, j, R);
            w = packed_set(w, j + 1, R);
        } else if (p > j + 1) {
            w = packed_set(w, p, R);
            w = packed_set(w, j, L);
            w = packed_set(w, j + 1, L);
        } else if (p < j && q > j + 1) {
            w = packed_set(w, j, R);
            w = packed_set(w, j + 1, L);
        } else {
            candidate = false;
        }
        if (candidate && packed_valid(w)) out.insert(w);
    }

    if (left > 0) {
        const int p = left - 1;
        if (packed_symbol(z, p) == L) {
            const int q = packed_partner(z, p);
            if (q == right && p < j && q > j + 1) {
                PackedWord w = packed_set(packed_set(z, j, R), j + 1, L);
                if (packed_valid(w)) out.insert(w);
            }
        }
    }

    const int root = packed_root(z);
    if (level == 0 || (left == 0 && right < W && root == right)) {
        if (root < j) {
            PackedWord w = packed_set(z, root, L);
            w = packed_set(w, j, R);
            w = packed_set(w, j + 1, R);
            if (packed_valid(w)) out.insert(w);
        } else if (root > j + 1) {
            PackedWord w = packed_set(packed_set(z, j, R), j + 1, L);
            if (packed_valid(w)) out.insert(w);
        }
    }
    return out;
}

SmallUnique<PackedKey, 2> packed_inverse_project(const PackedKey& dst, int j, int W) {
    SmallUnique<PackedKey, 2> out;
    out.insert(dst);
    if (dst.type == 'A' && j <= W - 3 && j + 1 < dst.w.len &&
        packed_symbol(dst.w, j) == L && packed_symbol(dst.w, j + 1) == R) {
        PackedWord z = packed_remove(dst.w, j + 1);
        z = packed_set(z, j, N);
        const PackedKey c{'C', z};
        if (!(packed_project(c, j, W) == dst)) fail("packed inverse project");
        out.insert(c);
    }
    return out;
}

bool packed_inverse_E(const PackedWord& z, int i, PackedKey& out) {
    const bool a = packed_symbol(z, i) != N;
    const bool b = packed_symbol(z, i + 1) != N;
    if (!a && !b) {
        out = PackedKey{'A', packed_remove(z, i + 1)};
        return true;
    }
    if (a != b) {
        out = PackedKey{'A', packed_collapse_A(z, i)};
        return true;
    }
    if (packed_symbol(z, i) == L && packed_symbol(z, i + 1) == R) {
        out = PackedKey{'C', packed_remove_pair(z, i)};
        return true;
    }
    return false;
}

bool packed_in_source_layout(const PackedKey& k, int W, int i) {
    if (k.type == 'A') return k.w.len == W - 1 && packed_valid(k.w);
    if (k.type == 'C')
        return k.w.len == W - 2 && packed_valid(k.w) &&
               (i > W - 3 || packed_symbol(k.w, i) != N);
    return false;
}

SmallUnique<PackedKey, 64> packed_inverse_K(const PackedKey& dst, int W, int i) {
    SmallUnique<PackedKey, 64> out;
    const auto raw = packed_inverse_project(dst, i + 1, W);
    for (int a = 0; a < raw.size; ++a) {
        const auto full = packed_inverse_R_raw(raw.value[a], i + 1, W);
        for (int b = 0; b < full.size; ++b) {
            PackedKey src;
            if (packed_inverse_E(full.value[b], i, src) &&
                packed_in_source_layout(src, W, i))
                out.insert(src);
        }
    }
    return out;
}

SmallUnique<PackedKey, 32> packed_component_sources(PackedKey seed, int W, int i) {
    SmallUnique<PackedKey, 32> sources;
    sources.insert(seed);
    // Exhaustive depth classification shows maximum bipartite distance seven:
    // three source->destination->source expansion rounds discover every source.
    for (int round = 0; round < 3; ++round) {
        const int old_size = sources.size;
        for (int s = 0; s < old_size; ++s) {
            const auto dst = packed_K(sources.value[s], W, i);
            for (int d = 0; d < dst.size; ++d) {
                const auto pre = packed_inverse_K(dst.value[d], W, i);
                for (int q = 0; q < pre.size; ++q) sources.insert(pre.value[q]);
            }
        }
    }
    return sources;
}

std::set<Key> oracle_component_sources(const Key& seed, int W, int i) {
    std::set<Key> sources{seed};
    std::deque<Key> queue{seed};
    while (!queue.empty()) {
        const Key src = queue.front();
        queue.pop_front();
        for (const auto& [dst, c] : K_basis(src, W, i)) {
            if (c != 1) fail("packed oracle nonunit");
            for (const Key& pre : inverse_K(dst, W, i))
                if (sources.insert(pre).second) queue.push_back(pre);
        }
    }
    return sources;
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 12;
    if (maxW < 4 || maxW > 15) return 2;

    std::vector<std::vector<Word>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 4; W <= maxW; ++W) {
        Rank checked_k = 0;
        Rank checked_inverse = 0;
        Rank checked_components = 0;
        Rank max_pairs = 0;
        Rank max_inverse = 0;

        for (int len = W - 2; len <= W - 1; ++len) {
            for (const Word& w : words[len]) {
                const PackedWord p = pack_word(w);
                if (unpack_word(p) != w || !packed_valid(p))
                    fail("packed word roundtrip W=" + std::to_string(W));
                for (int pos = 0; pos < len; ++pos) {
                    if (w[pos] == N) continue;
                    const int a = packed_partner(p, pos);
                    const int b = partner(w, pos);
                    if (a != b)
                        fail("packed partner W=" + std::to_string(W));
                }
            }
        }

        for (int i = 0; i <= W - 4; ++i) {
            const auto src = q_basis(W, i, words);
            const auto dst = q_basis(W, i + 1, words);
            for (const Key& k : src) {
                const CVec expected = K_basis(k, W, i);
                const auto got = packed_K(pack_key(k), W, i);
                CVec actual;
                for (int q = 0; q < got.size; ++q) add(actual, unpack_key(got.value[q]));
                if (actual != expected)
                    fail("packed K mismatch W=" + std::to_string(W) +
                         " i=" + std::to_string(i));
                ++checked_k;
            }

            for (const Key& k : dst) {
                const auto expected = inverse_K(k, W, i);
                const auto got = packed_inverse_K(pack_key(k), W, i);
                std::set<Key> actual;
                for (int q = 0; q < got.size; ++q) actual.insert(unpack_key(got.value[q]));
                if (actual != expected)
                    fail("packed inverse mismatch W=" + std::to_string(W) +
                         " i=" + std::to_string(i));
                max_inverse = std::max<Rank>(max_inverse, got.size);
                ++checked_inverse;
            }

            for (const Word& u : words[W - 2]) {
                const Key seed = project_key(Key{'C', u}, i, W);
                const auto expected = oracle_component_sources(seed, W, i);
                const auto got = packed_component_sources(pack_key(seed), W, i);
                std::set<Key> actual;
                for (int q = 0; q < got.size; ++q) actual.insert(unpack_key(got.value[q]));
                if (actual != expected)
                    fail("packed component mismatch W=" + std::to_string(W) +
                         " i=" + std::to_string(i));

                // Fixed-round reconstruction must already be closed.
                for (int q = 0; q < got.size; ++q) {
                    const auto ds = packed_K(got.value[q], W, i);
                    for (int d = 0; d < ds.size; ++d) {
                        const auto pre = packed_inverse_K(ds.value[d], W, i);
                        for (int p = 0; p < pre.size; ++p)
                            if (!got.contains(pre.value[p]))
                                fail("packed component not closed W=" + std::to_string(W));
                    }
                }
                max_pairs = std::max<Rank>(max_pairs, got.size);
                ++checked_components;
            }
        }

        const Rank observed_bound = Rank(W / 2 + 3);
        if (max_pairs > observed_bound)
            fail("packed component exceeds observed half-width bound W=" + std::to_string(W));

        std::cout << "W=" << W
                  << " packed_words=2xu32"
                  << " checked_K=" << checked_k
                  << " checked_inverse=" << checked_inverse
                  << " checked_components=" << checked_components
                  << " max_inverse=" << max_inverse
                  << " max_pairs=" << max_pairs
                  << " observed_pair_bound=floor(W/2)+3"
                  << " expansion_rounds=3"
                  << " strings_hotpath=0 mate_array=0 component_table=0"
                  << " OK\n";
    }

    std::cout << "W=28_layout support_bits=27 primitive_bits=27"
              << " packed_state_bytes=8"
              << " component_local_capacity_32=1"
              << "\n";
    std::cout << "ALL_OK packed_component_fixed_round=1\n";
    return 0;
}
