#pragma push_macro("main")
#undef main
#define main two_cell_direct_component_probe_main_unused
#include "two_cell_direct_component_probe.cpp"
#pragma pop_macro("main")

#include <numeric>

namespace {

struct DSU {
    std::vector<int> p, sz;
    explicit DSU(int n) : p(static_cast<std::size_t>(n)), sz(static_cast<std::size_t>(n), 1) {
        std::iota(p.begin(), p.end(), 0);
    }
    int find(int x) {
        while (p[x] != x) {
            p[x] = p[p[x]];
            x = p[x];
        }
        return x;
    }
    void join(int a, int b) {
        a = find(a); b = find(b);
        if (a == b) return;
        if (sz[a] < sz[b]) std::swap(a, b);
        p[b] = a; sz[a] += sz[b];
    }
};

Rank catalan(int n) {
    // Small-width probe/theory helper. W=28 only needs Catalan(14).
    Rank c = 1;
    for (int k = 0; k < n; ++k)
        c = c * Rank(2 * (2 * k + 1)) / Rank(k + 2);
    return c;
}

Rank primitive_count(int occupied) {
    if (occupied <= 0 || !(occupied & 1)) return 0;
    return catalan((occupied + 1) / 2);
}

Rank binom_small(int n, int k) {
    if (k < 0 || k > n) return 0;
    Rank z = 1;
    for (int j = 1; j <= k; ++j)
        z = z * Rank(n - k + j) / Rank(j);
    return z;
}

Rank label_block_size(int k) {
    Rank z = 0;
    for (int l = 0; l <= 3; ++l)
        z += binom_small(3, l) * primitive_count(k + l);
    return z;
}

Rank state_block_size(int k) {
    Rank a = 0, c = 0;
    // A state: four local support bits around the two adjacent reduced cuts.
    for (int l = 0; l <= 4; ++l)
        a += binom_small(4, l) * primitive_count(k + l);
    // C state in Q_{i+1}: the middle local support bit is fixed occupied.
    for (int r = 0; r <= 2; ++r)
        c += binom_small(2, r) * primitive_count(k + 1 + r);
    return a + c;
}

std::uint64_t outer_support_mask(const Word& u, int i) {
    std::uint64_t mask = 0;
    int q = 0;
    for (int p = 0; p < static_cast<int>(u.size()); ++p) {
        if (p == i || p == i + 1 || p == i + 2) continue;
        if (u[p] != N) mask |= std::uint64_t(1) << q;
        ++q;
    }
    return mask;
}

int popcount64(std::uint64_t x) {
#if defined(__GNUG__) || defined(__clang__)
    return __builtin_popcountll(x);
#else
    int n = 0;
    while (x) { x &= x - 1; ++n; }
    return n;
#endif
}

struct RecouplingStats {
    Rank states = 0;
    Rank labels = 0;
    Rank blocks = 0;
    Rank max_label_block = 0;
    Rank max_state_block = 0;
    int max_symbol_diffs = 0;
    int max_outer_orientation_diffs = 0;
};

RecouplingStats verify_recoupling(
    int W,
    int i,
    const std::vector<std::vector<Word>>& words
) {
    assert(i >= 0 && i + 1 <= W - 4);
    const auto& labels = words[W - 2];
    const int m = static_cast<int>(labels.size());

    std::map<Key, Word> left_label;
    std::map<Key, Word> right_label;

    // Destination partition of K_i, indexed by the component label on its left.
    for (const Word& u : labels) {
        const auto src = packed_direct_component_sources(pack_word(u), W, i);
        for (int q = 0; q < src.size; ++q) {
            const auto dst = packed_K(src.value[q], W, i);
            for (int d = 0; d < dst.size; ++d) {
                const Key key = unpack_key(dst.value[d]);
                const auto [it, inserted] = left_label.emplace(key, u);
                if (!inserted && it->second != u)
                    fail("recoupling destination belongs to two left components");
            }
        }
    }

    // Source partition of K_{i+1}, indexed by the component label on its right.
    for (const Word& v : labels) {
        const auto src = packed_direct_component_sources(pack_word(v), W, i + 1);
        for (int q = 0; q < src.size; ++q) {
            const Key key = unpack_key(src.value[q]);
            const auto [it, inserted] = right_label.emplace(key, v);
            if (!inserted && it->second != v)
                fail("recoupling state belongs to two right components");
        }
    }

    if (left_label.size() != right_label.size() || left_label.size() == 0)
        fail("recoupling state-set dimension");
    for (const auto& [key, u] : left_label)
        if (!right_label.count(key)) fail("recoupling state-set mismatch");

    std::map<Word, int> label_rank;
    for (int r = 0; r < m; ++r) label_rank.emplace(labels[r], r);
    DSU dsu(2 * m);

    std::map<std::uint64_t, Rank> left_count;
    std::map<std::uint64_t, Rank> right_count;
    std::map<std::uint64_t, Rank> state_count;
    RecouplingStats st;
    st.states = left_label.size();
    st.labels = labels.size();

    for (const auto& [key, u] : left_label) {
        const Word& v = right_label.at(key);
        const std::uint64_t lm = outer_support_mask(u, i);
        const std::uint64_t rm = outer_support_mask(v, i);
        if (lm != rm)
            fail("recoupling changed outer support W=" + std::to_string(W) +
                 " i=" + std::to_string(i));

        int total_diff = 0;
        int outer_diff = 0;
        for (int p = 0; p < W - 2; ++p) {
            if (u[p] == v[p]) continue;
            ++total_diff;
            if (p < i || p > i + 2) {
                ++outer_diff;
                if (u[p] == N || v[p] == N)
                    fail("recoupling outer orientation changed occupancy");
            }
        }
        st.max_symbol_diffs = std::max(st.max_symbol_diffs, total_diff);
        st.max_outer_orientation_diffs =
            std::max(st.max_outer_orientation_diffs, outer_diff);
        if (total_diff > 4)
            fail("recoupling exceeds four symbol edits");

        const int a = label_rank.at(u);
        const int b = label_rank.at(v);
        dsu.join(a, m + b);
        ++state_count[lm];
    }

    for (const Word& u : labels) ++left_count[outer_support_mask(u, i)];
    for (const Word& v : labels) ++right_count[outer_support_mask(v, i)];

    const Rank expected_blocks = Rank(1) << (W - 5);
    if (left_count.size() != expected_blocks || right_count.size() != expected_blocks ||
        state_count.size() != expected_blocks)
        fail("recoupling outer-mask count");

    std::set<int> roots;
    std::map<std::uint64_t, int> mask_root;
    for (int r = 0; r < m; ++r) {
        const std::uint64_t mask = outer_support_mask(labels[r], i);
        const int lr = dsu.find(r);
        const int rr = dsu.find(m + r);
        roots.insert(lr);
        roots.insert(rr);
        auto [it, inserted] = mask_root.emplace(mask, lr);
        if (!inserted && dsu.find(it->second) != lr)
            fail("recoupling mask split into multiple components");
        if (dsu.find(it->second) != rr)
            fail("recoupling left/right mask disconnected");
    }

    std::set<int> canonical_roots;
    for (auto& [mask, r] : mask_root) canonical_roots.insert(dsu.find(r));
    if (canonical_roots.size() != expected_blocks)
        fail("recoupling connected-component formula");
    st.blocks = expected_blocks;

    for (const auto& [mask, lc] : left_count) {
        const int k = popcount64(mask);
        const Rank expect_labels = label_block_size(k);
        const Rank expect_states = state_block_size(k);
        if (lc != expect_labels || right_count.at(mask) != expect_labels ||
            state_count.at(mask) != expect_states)
            fail("recoupling popcount block formula W=" + std::to_string(W) +
                 " i=" + std::to_string(i) + " k=" + std::to_string(k));
        st.max_label_block = std::max(st.max_label_block, lc);
        st.max_state_block = std::max(st.max_state_block, state_count.at(mask));
    }

    Rank total_labels = 0, total_states = 0;
    const int q = W - 5;
    for (int k = 0; k <= q; ++k) {
        total_labels += binom_small(q, k) * label_block_size(k);
        total_states += binom_small(q, k) * state_block_size(k);
    }
    if (total_labels != labels.size() || total_states != left_label.size())
        fail("recoupling total block identity");

    return st;
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 11;
    if (maxW < 6 || maxW > 14) return 2;

    std::vector<std::vector<Word>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 6; W <= maxW; ++W) {
        RecouplingStats worst;
        for (int i = 0; i <= W - 5; ++i) {
            const RecouplingStats s = verify_recoupling(W, i, words);
            if (!worst.states) worst = s;
            if (s.states != worst.states || s.labels != worst.labels || s.blocks != worst.blocks)
                fail("recoupling position-dependent totals");
            worst.max_label_block = std::max(worst.max_label_block, s.max_label_block);
            worst.max_state_block = std::max(worst.max_state_block, s.max_state_block);
            worst.max_symbol_diffs = std::max(worst.max_symbol_diffs, s.max_symbol_diffs);
            worst.max_outer_orientation_diffs =
                std::max(worst.max_outer_orientation_diffs, s.max_outer_orientation_diffs);
        }
        std::cout << "W=" << W
                  << " labels=" << worst.labels
                  << " states=" << worst.states
                  << " outer_support_blocks=" << worst.blocks
                  << " expected_blocks=2^(W-5)"
                  << " max_label_block=" << worst.max_label_block
                  << " max_state_block=" << worst.max_state_block
                  << " max_symbol_diffs=" << worst.max_symbol_diffs
                  << " max_outer_orientation_diffs=" << worst.max_outer_orientation_diffs
                  << " support_outside_local3_invariant=1"
                  << " block_size_depends_only_on_popcount=1"
                  << " OK\n";
    }

    const int W = 28;
    const int q = W - 5;
    Rank max_b = 0, max_e = 0;
    int max_b_k = -1, max_e_k = -1;
    Rank total_b = 0, total_e = 0;
    for (int k = 0; k <= q; ++k) {
        const Rank b = label_block_size(k);
        const Rank e = state_block_size(k);
        total_b += binom_small(q, k) * b;
        total_e += binom_small(q, k) * e;
        if (b > max_b) { max_b = b; max_b_k = k; }
        if (e > max_e) { max_e = e; max_e_k = k; }
    }
    std::cout << "W=28_theory outer_support_bits=" << q
              << " blocks=" << (Rank(1) << q)
              << " labels=" << total_b
              << " states=" << total_e
              << " max_label_block=" << max_b
              << " max_label_block_k=" << max_b_k
              << " max_state_block=" << max_e
              << " max_state_block_k=" << max_e_k
              << " max_state_block_u32_MiB="
              << double(max_e) * 4.0 / double(1ULL << 20)
              << "\n";
    std::cout << "ALL_OK recoupling_outer_support_factorization=1\n";
    return 0;
}
