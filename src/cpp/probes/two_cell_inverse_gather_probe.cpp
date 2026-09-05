#pragma push_macro("main")
#undef main
#define main two_cell_channel_probe_main_unused
#include "two_cell_channel_probe.cpp"
#pragma pop_macro("main")

#include <optional>

namespace {

std::vector<int> path_heights(const Word& w) {
    std::vector<int> h(w.size() + 1);
    h[0] = 1;
    for (std::size_t i = 0; i < w.size(); ++i) {
        h[i + 1] = h[i];
        if (w[i] == L) ++h[i + 1];
        else if (w[i] == R) --h[i + 1];
    }
    return h;
}

Word split_arc_candidate(const Word& z, int j, int p, int q) {
    Word w = z;
    if (q < j) {
        w[q] = L;
        w[j] = R;
        w[j + 1] = R;
    } else if (p > j + 1) {
        w[p] = R;
        w[j] = L;
        w[j + 1] = L;
    } else if (p < j && q > j + 1) {
        w[j] = R;
        w[j + 1] = L;
    } else {
        return {};
    }
    return valid_word(w) ? w : Word{};
}

Word split_root_candidate(const Word& z, int j, int root) {
    Word w = z;
    if (root < j) {
        w[root] = L;
        w[j] = R;
        w[j + 1] = R;
    } else if (root > j + 1) {
        w[j] = R;
        w[j + 1] = L;
    } else {
        return {};
    }
    return valid_word(w) ? w : Word{};
}

std::set<Word> inverse_R_raw(const Key& raw, int j, int W) {
    std::set<Word> out;
    if (raw.type == 'C') {
        const Word w = raw.w.substr(0, j) + Word(2, N) + raw.w.substr(j);
        if (static_cast<int>(w.size()) != W || !valid_word(w)) std::abort();
        out.insert(w);
        return out;
    }

    assert(raw.type == 'A');
    assert(static_cast<int>(raw.w.size()) == W - 1);
    const char local = raw.w[j];
    if (local != N) {
        const Word a = raw.w.substr(0, j) + local + N + raw.w.substr(j + 1);
        const Word b = raw.w.substr(0, j) + N + local + raw.w.substr(j + 1);
        if (valid_word(a)) out.insert(a);
        if (valid_word(b)) out.insert(b);
        return out;
    }

    // Expand the collapsed vacancy back to the two vacant sites seen after a
    // cap. Every nontrivial inverse is obtained by cutting one strand on the
    // boundary of the face containing the gap j,j+1.
    const Word z = raw.w.substr(0, j) + Word(2, N) + raw.w.substr(j + 1);
    if (static_cast<int>(z.size()) != W || !valid_word(z)) std::abort();
    out.insert(z); // vacancy -> vacancy

    const LinkState state = decode(z);
    const std::vector<int> h = path_heights(z);
    const int level = h[j];

    // Maximal level-face interval containing the two vacancies. Vertices in
    // [left,right] stay at height >= level. Excursions that start at exactly
    // level are the sibling strands visible from this face.
    int left = j;
    while (left > 0 && h[left - 1] >= level) --left;
    int right = j + 2;
    while (right < W && h[right + 1] >= level) ++right;

    for (int p = 0; p < W; ++p) {
        const int q = state.mate[p];
        if (q <= p) continue;
        if (p >= left && q < right && h[p] == level) {
            const Word w = split_arc_candidate(z, j, p, q);
            if (!w.empty()) out.insert(w);
        }
    }

    // If the face is enclosed, its innermost enclosing arc is another legal
    // strand to cut. Cutting any outer enclosing arc would encode the same
    // local RL word, so only the face boundary is needed.
    if (left > 0) {
        const int p = left - 1;
        const int q = state.mate[p];
        if (z[p] == L && q == right && p < j && q > j + 1) {
            const Word w = split_arc_candidate(z, j, p, q);
            if (!w.empty()) out.insert(w);
        }
    }

    // The distinguished root strand bounds the level-1 prefix face and is an
    // ordinary exposed strand on the outer level-0 face.
    if (level == 0 || (left == 0 && right < W && state.root == right)) {
        const Word w = split_root_candidate(z, j, state.root);
        if (!w.empty()) out.insert(w);
    }

    return out;
}

std::vector<Key> inverse_project(const Key& dst, int j, int W) {
    std::vector<Key> out{dst};
    if (dst.type == 'A' && j <= W - 3 && j + 1 < static_cast<int>(dst.w.size()) &&
        dst.w[j] == L && dst.w[j + 1] == R) {
        const Key c{'C', dst.w.substr(0, j) + N + dst.w.substr(j + 2)};
        if (!(project_key(c, j, W) == dst)) std::abort();
        out.push_back(c);
    }
    return out;
}

std::optional<Key> inverse_E(const Word& z, int i) {
    const bool a = z[i] != N, b = z[i + 1] != N;
    if (!a && !b)
        return Key{'A', z.substr(0, i) + N + z.substr(i + 2)};
    if (a != b) {
        const char s = a ? z[i] : z[i + 1];
        return Key{'A', z.substr(0, i) + s + z.substr(i + 2)};
    }
    if (z[i] == L && z[i + 1] == R)
        return Key{'C', z.substr(0, i) + z.substr(i + 2)};
    return std::nullopt;
}

bool in_source_layout(const Key& k, int W, int i) {
    if (k.type == 'A')
        return static_cast<int>(k.w.size()) == W - 1 && valid_word(k.w);
    if (k.type == 'C')
        return static_cast<int>(k.w.size()) == W - 2 && valid_word(k.w) &&
               (i > W - 3 || k.w[i] != N);
    return false;
}

std::set<Key> inverse_K(const Key& dst, int W, int i) {
    const int j = i + 1;
    std::set<Key> out;
    for (const Key& raw : inverse_project(dst, j, W)) {
        for (const Word& z : inverse_R_raw(raw, j, W)) {
            const auto src = inverse_E(z, i);
            if (src && in_source_layout(*src, W, i)) out.insert(*src);
        }
    }
    return out;
}

// Rank/unrank codec for the canonical reduced layout. `fixed` forbids N at
// one position; this is exactly the canonical C block condition. The DP is
// O(W^2) words and replaces the dense Key->rank maps used by the oracle.
struct WordRankCodec {
    int len = 0;
    int fixed = -1;
    std::vector<std::vector<Rank>> dp;

    WordRankCodec(int len_, int fixed_ = -1)
        : len(len_), fixed(fixed_),
          dp(static_cast<std::size_t>(len_ + 1),
             std::vector<Rank>(static_cast<std::size_t>(len_ + 2), 0)) {
        dp[len][0] = 1;
        for (int pos = len - 1; pos >= 0; --pos) {
            for (int h = 0; h <= len; ++h) {
                Rank z = 0;
                if (pos != fixed) z += dp[pos + 1][h];
                if (h > 0) z += dp[pos + 1][h - 1];
                z += dp[pos + 1][h + 1];
                dp[pos][h] = z;
            }
        }
    }

    Rank size() const { return dp[0][1]; }
    std::size_t logical_bytes() const {
        return std::size_t(len + 1) * std::size_t(len + 2) * sizeof(Rank);
    }

    static int order(char c) { return c == N ? 0 : (c == R ? 1 : 2); }

    Rank rank(const Word& w) const {
        assert(static_cast<int>(w.size()) == len);
        if (fixed >= 0) assert(w[fixed] != N);
        Rank r = 0;
        int h = 1;
        const char options[3] = {N, R, L};
        for (int pos = 0; pos < len; ++pos) {
            for (char x : options) {
                if (pos == fixed && x == N) continue;
                if (order(x) >= order(w[pos])) break;
                if (x == R && h == 0) continue;
                const int nh = h + (x == L ? 1 : (x == R ? -1 : 0));
                if (nh >= 0 && nh < static_cast<int>(dp[pos + 1].size()))
                    r += dp[pos + 1][nh];
            }
            h += w[pos] == L ? 1 : (w[pos] == R ? -1 : 0);
            assert(h >= 0);
        }
        assert(h == 0);
        return r;
    }

    Word unrank(Rank r) const {
        assert(r < size());
        Word w;
        w.reserve(static_cast<std::size_t>(len));
        int h = 1;
        const char options[3] = {N, R, L};
        for (int pos = 0; pos < len; ++pos) {
            bool found = false;
            for (char x : options) {
                if (pos == fixed && x == N) continue;
                if (x == R && h == 0) continue;
                const int nh = h + (x == L ? 1 : (x == R ? -1 : 0));
                Rank z = 0;
                if (nh >= 0 && nh < static_cast<int>(dp[pos + 1].size()))
                    z = dp[pos + 1][nh];
                if (r < z) {
                    w.push_back(x);
                    h = nh;
                    found = true;
                    break;
                }
                r -= z;
            }
            assert(found);
        }
        assert(h == 0 && valid_word(w));
        return w;
    }
};

struct ReducedRankCodec {
    int W = 0;
    int fixed = 0;
    WordRankCodec a;
    WordRankCodec c;

    ReducedRankCodec(int W_, int fixed_)
        : W(W_), fixed(fixed_), a(W_ - 1), c(W_ - 2, fixed_) {}

    Rank size() const { return a.size() + c.size(); }
    std::size_t logical_bytes() const { return a.logical_bytes() + c.logical_bytes(); }

    Rank rank(const Key& k) const {
        if (k.type == 'A') return a.rank(k.w);
        assert(k.type == 'C');
        return a.size() + c.rank(k.w);
    }

    Key unrank(Rank r) const {
        if (r < a.size()) return Key{'A', a.unrank(r)};
        return Key{'C', c.unrank(r - a.size())};
    }
};

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::stoi(argv[1]) : 12;
    if (maxW < 4 || maxW > 15) {
        std::cerr << "maxW must be 4..15\n";
        return 2;
    }

    std::vector<std::vector<Word>> words(maxW + 1);
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 4; W <= maxW; ++W) {
        const Rank m1 = words[W - 1].size();
        const Rank m2 = words[W - 2].size();
        const Rank m3 = words[W - 3].size();
        const Rank dim = m1 + m2 - m3;
        const Rank nnz = 2 * m1 + m2 - 2 * m3;
        Rank max_indeg = 0;
        Rank max_c_indeg = 0;
        std::size_t max_unique_codec_bytes = 0;

        for (int i = 0; i <= W - 4; ++i) {
            const ReducedLayout src = make_layout(W, i, words);
            const ReducedLayout dst = make_layout(W, i + 1, words);
            const ReducedRankCodec src_codec(W, i);
            const ReducedRankCodec dst_codec(W, i + 1);
            if (src.size() != dim || dst.size() != dim ||
                src_codec.size() != dim || dst_codec.size() != dim)
                std::abort();

            // The A codec is identical for source and destination, so a
            // production step only needs A + C_i + C_{i+1} count tables.
            max_unique_codec_bytes = std::max(
                max_unique_codec_bytes,
                src_codec.a.logical_bytes() + src_codec.c.logical_bytes() +
                    dst_codec.c.logical_bytes());

            for (Rank r = 0; r < dim; ++r) {
                if (!(src_codec.unrank(r) == src.key[r]) || src_codec.rank(src.key[r]) != r)
                    fail("source codec mismatch W=" + std::to_string(W) +
                         " i=" + std::to_string(i) + " r=" + std::to_string(r));
                if (!(dst_codec.unrank(r) == dst.key[r]) || dst_codec.rank(dst.key[r]) != r)
                    fail("destination codec mismatch W=" + std::to_string(W) +
                         " i=" + std::to_string(i) + " r=" + std::to_string(r));
            }

            std::vector<std::vector<Rank>> incoming(static_cast<std::size_t>(dim));
            for (Rank s = 0; s < dim; ++s) {
                if (src_codec.rank(src.key[s]) != s) std::abort();
                for (const auto& [k, c] : K_basis(src.key[s], W, i)) {
                    if (c != 1) std::abort();
                    incoming[dst_codec.rank(k)].push_back(s);
                }
            }

            std::vector<std::uint64_t> value(static_cast<std::size_t>(dim));
            std::vector<std::uint64_t> scatter(static_cast<std::size_t>(dim));
            std::vector<std::uint64_t> gather(static_cast<std::size_t>(dim));
            for (Rank s = 0; s < dim; ++s)
                value[s] = 1 + ((s * 0x9e3779b97f4a7c15ULL) ^
                                (Rank(W) << 32) ^ Rank(i));

            for (Rank d = 0; d < dim; ++d) {
                const Key dst_key = dst_codec.unrank(d);
                std::vector<Rank> got;
                for (const Key& k : inverse_K(dst_key, W, i))
                    got.push_back(src_codec.rank(k));
                std::sort(got.begin(), got.end());
                if (got != incoming[d])
                    fail("rank-table-free inverse mismatch W=" + std::to_string(W) +
                         " i=" + std::to_string(i) + " d=" + std::to_string(d));

                max_indeg = std::max<Rank>(max_indeg, got.size());
                if (dst_key.type == 'C') {
                    max_c_indeg = std::max<Rank>(max_c_indeg, got.size());
                    if (got.size() != 1)
                        fail("C destination must have one preimage W=" + std::to_string(W));
                }

                for (Rank s : incoming[d]) scatter[d] += value[s];
                for (Rank s : got) gather[d] += value[s];
            }
            if (scatter != gather)
                fail("rank-table-free gather mismatch W=" + std::to_string(W) +
                     " i=" + std::to_string(i));
        }

        const Rank csr_bytes = (dim + 1 + nnz) * sizeof(Rank);
        std::cout << "W=" << W
                  << " reduced=" << dim
                  << " nnz=" << nnz
                  << " max_indeg=" << max_indeg
                  << " max_c_indeg=" << max_c_indeg
                  << " csr_kib=" << double(csr_bytes) / 1024.0
                  << " inverse_table_bytes=0"
                  << " rank_lookup_bytes=0"
                  << " codec_kib=" << double(max_unique_codec_bytes) / 1024.0
                  << " inverse_scan=O(W)"
                  << " OK\n";
    }

    std::cout << "ALL_OK maxW=" << maxW
              << " table_free_inverse=1 rank_table_free=1\n";
    return 0;
}
