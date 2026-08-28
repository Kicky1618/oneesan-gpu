#pragma push_macro("main")
#undef main
#define main two_cell_channel_probe_main_unused
#include "two_cell_channel_probe.cpp"
#pragma pop_macro("main")

#include <deque>
#include <limits>

namespace {

using EdgeList = std::vector<std::vector<Rank>>;

std::uint64_t count_words_u64(int W) {
    std::vector<std::uint64_t> cur(static_cast<std::size_t>(W + 2), 0), next(cur.size());
    cur[1] = 1;
    for (int pos = 0; pos < W; ++pos) {
        std::fill(next.begin(), next.end(), 0);
        for (int h = 0; h <= W; ++h) {
            const auto x = cur[h];
            if (!x) continue;
            next[h] += x;
            next[h + 1] += x;
            if (h) next[h - 1] += x;
        }
        cur.swap(next);
    }
    return cur[0];
}

struct ForestStats {
    Rank edges = 0;
    Rank components = 0;
    Rank max_source_component = 0;
    Rank forced_matches = 0;
};

ForestStats verify_forest_step(
    int W,
    int i,
    const std::vector<std::vector<Word>>& words
) {
    const ReducedLayout src = make_layout(W, i, words);
    const ReducedLayout dst = make_layout(W, i + 1, words);
    if (src.size() != dst.size()) fail("forest dimension mismatch");

    const Rank n = src.size();
    EdgeList out(static_cast<std::size_t>(n));
    EdgeList in(static_cast<std::size_t>(n));
    Rank edges = 0;
    for (Rank s = 0; s < n; ++s) {
        const CVec col = K_basis(src.key[s], W, i);
        for (const auto& [k, c] : col) {
            if (c != 1) fail("forest nonunit coefficient");
            const auto it = dst.rank.find(k);
            if (it == dst.rank.end()) fail("forest destination missing");
            out[s].push_back(it->second);
            in[it->second].push_back(s);
            ++edges;
        }
    }

    const Rank m1 = static_cast<Rank>(words[W - 1].size());
    const Rank m2 = static_cast<Rank>(words[W - 2].size());
    const Rank m3 = static_cast<Rank>(words[W - 3].size());
    const Rank expected_n = m1 + m2 - m3;
    const Rank expected_edges = 2 * m1 + m2 - 2 * m3;
    if (n != expected_n || edges != expected_edges) fail("forest formula mismatch");

    std::vector<std::uint8_t> seen_l(static_cast<std::size_t>(n), 0);
    std::vector<std::uint8_t> seen_r(static_cast<std::size_t>(n), 0);
    Rank components = 0;
    Rank max_source_component = 0;

    struct Vertex { bool right; Rank id; };
    for (Rank start = 0; start < 2 * n; ++start) {
        const bool start_right = start >= n;
        const Rank sid = start_right ? start - n : start;
        auto& seen = start_right ? seen_r : seen_l;
        if (seen[sid]) continue;

        ++components;
        Rank left_count = 0, right_count = 0, degree_sum = 0;
        std::deque<Vertex> q;
        seen[sid] = 1;
        q.push_back({start_right, sid});
        while (!q.empty()) {
            const Vertex v = q.front();
            q.pop_front();
            if (v.right) {
                ++right_count;
                degree_sum += static_cast<Rank>(in[v.id].size());
                for (Rank u : in[v.id]) {
                    if (!seen_l[u]) {
                        seen_l[u] = 1;
                        q.push_back({false, u});
                    }
                }
            } else {
                ++left_count;
                degree_sum += static_cast<Rank>(out[v.id].size());
                for (Rank d : out[v.id]) {
                    if (!seen_r[d]) {
                        seen_r[d] = 1;
                        q.push_back({true, d});
                    }
                }
            }
        }
        if (left_count != right_count) fail("forest unbalanced component");
        const Rank component_edges = degree_sum / 2;
        if (component_edges + 1 != left_count + right_count)
            fail("reduced transition component is not a tree");
        max_source_component = std::max(max_source_component, left_count);
    }

    if (components != m2) fail("forest component count is not M_{W-2}");
    if (max_source_component > Rank(W / 2 + 3))
        fail("forest component exceeded observed linear bound");

    // Every degree-one edge is forced in any perfect matching. Repeatedly
    // peel forced source/destination leaves. If all vertices disappear, the
    // perfect matching exists and is unique without running a matching search.
    std::vector<Rank> ldeg(static_cast<std::size_t>(n));
    std::vector<Rank> rdeg(static_cast<std::size_t>(n));
    std::vector<std::uint8_t> live_l(static_cast<std::size_t>(n), 1);
    std::vector<std::uint8_t> live_r(static_cast<std::size_t>(n), 1);
    std::deque<Vertex> leaves;
    for (Rank u = 0; u < n; ++u) {
        ldeg[u] = static_cast<Rank>(out[u].size());
        rdeg[u] = static_cast<Rank>(in[u].size());
        if (ldeg[u] == 1) leaves.push_back({false, u});
        if (rdeg[u] == 1) leaves.push_back({true, u});
    }

    auto only_live_right = [&](Rank u) -> Rank {
        for (Rank v : out[u]) if (live_r[v]) return v;
        return std::numeric_limits<Rank>::max();
    };
    auto only_live_left = [&](Rank v) -> Rank {
        for (Rank u : in[v]) if (live_l[u]) return u;
        return std::numeric_limits<Rank>::max();
    };

    Rank matched = 0;
    while (!leaves.empty()) {
        const Vertex x = leaves.front();
        leaves.pop_front();
        Rank u = 0, v = 0;
        if (x.right) {
            v = x.id;
            if (!live_r[v] || rdeg[v] != 1) continue;
            u = only_live_left(v);
            if (u == std::numeric_limits<Rank>::max() || !live_l[u]) continue;
        } else {
            u = x.id;
            if (!live_l[u] || ldeg[u] != 1) continue;
            v = only_live_right(u);
            if (v == std::numeric_limits<Rank>::max() || !live_r[v]) continue;
        }

        live_l[u] = 0;
        live_r[v] = 0;
        ++matched;
        for (Rank vv : out[u]) {
            if (!live_r[vv]) continue;
            if (rdeg[vv] == 0) fail("forest right degree underflow");
            --rdeg[vv];
            if (rdeg[vv] == 1) leaves.push_back({true, vv});
        }
        for (Rank uu : in[v]) {
            if (!live_l[uu]) continue;
            if (ldeg[uu] == 0) fail("forest left degree underflow");
            --ldeg[uu];
            if (ldeg[uu] == 1) leaves.push_back({false, uu});
        }
    }
    if (matched != n) fail("forest leaf peeling did not produce full matching");

    return ForestStats{edges, components, max_source_component, matched};
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 14;
    if (maxW < 4 || maxW > 14) {
        std::cerr << "maxW must be 4..14\n";
        return 2;
    }

    std::vector<std::vector<Word>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 4; W <= maxW; ++W) {
        Rank global_max_component = 0;
        Rank expected_edges = 0;
        for (int i = 0; i <= W - 4; ++i) {
            const ForestStats s = verify_forest_step(W, i, words);
            global_max_component = std::max(global_max_component, s.max_source_component);
            expected_edges = s.edges;
        }
        const Rank m1 = static_cast<Rank>(words[W - 1].size());
        const Rank m2 = static_cast<Rank>(words[W - 2].size());
        const Rank m3 = static_cast<Rank>(words[W - 3].size());
        const Rank dim = m1 + m2 - m3;
        std::cout << "W=" << W
                  << " dim=" << dim
                  << " edges=" << expected_edges
                  << " components=" << m2
                  << " max_component=" << global_max_component
                  << " forest=1 unique_permutation=1 leaf_peeling=1\n";
    }

    constexpr int W = 28;
    const std::uint64_t m1 = count_words_u64(W - 1);
    const std::uint64_t m2 = count_words_u64(W - 2);
    const std::uint64_t m3 = count_words_u64(W - 3);
    const std::uint64_t dim = m1 + m2 - m3;
    const std::uint64_t edges = 2 * m1 + m2 - 2 * m3;
    std::cout << "W28_projection"
              << " full=" << count_words_u64(W)
              << " reduced=" << dim
              << " components=" << m2
              << " edges=" << edges
              << " extra_edges=" << (edges - dim)
              << " avg_states_per_tree=" << (double(dim) / double(m2))
              << " avg_edges_per_tree=" << (double(edges) / double(m2))
              << " max_component_bound=" << (W / 2 + 3)
              << '\n';
    return 0;
}
