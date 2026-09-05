#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_channel_probe_main_unused
#include "gridfp_reduced_production_channel_probe.cpp"
#pragma pop_macro("main")

#include <deque>
#include <set>

namespace {

struct ComponentStats {
    Rank components = 0;
    Rank edges = 0;
    Rank max_pairs = 0;
    Rank max_edges = 0;
    Rank max_cyclomatic = 0;
};

std::vector<Key> component_seeds(
    const std::vector<MateID>& labels,
    int W,
    int p,
    bool reverse
) {
    std::vector<Key> out;
    for (MateID v : labels) {
        const int other = reverse ? p : p - 2;
        if (mget(v, p - 1) == N && mget(v, other) == N) continue;
        if (mget(v, p - 1) != N) {
            out.push_back(Key{true, v});
        } else {
            const MateID m = reverse
                ? blocked_exclude_reverse(v, W, p)
                : blocked_exclude(v, p);
            out.push_back(Key{false, m});
        }
    }
    return out;
}

ComponentStats verify_components(
    const std::vector<MateID>& main,
    const std::vector<MateID>& block,
    const std::vector<MateID>& labels,
    int W,
    int p,
    bool reverse
) {
    const int next = reverse ? p + 1 : p - 1;
    const std::vector<Key> src = layout(main, block, p);
    const std::vector<Key> dst = layout(main, block, next);
    if (src.size() != dst.size()) fail("component square layout");
    const Rank n = src.size();

    std::map<Key, Rank> src_rank, dst_rank;
    for (Rank r = 0; r < n; ++r) {
        src_rank.emplace(src[static_cast<std::size_t>(r)], r);
        dst_rank.emplace(dst[static_cast<std::size_t>(r)], r);
    }

    std::vector<std::vector<Rank>> adj(static_cast<std::size_t>(2 * n));
    Rank edges = 0;
    for (Rank s = 0; s < n; ++s) {
        const Vec col = reduced_step_basis(src[static_cast<std::size_t>(s)], W, p, reverse);
        for (const auto& [z, c] : col) {
            if (c != 1 && c != -1) fail("component coefficient");
            const auto it = dst_rank.find(z);
            if (it == dst_rank.end()) fail("component destination rank");
            const Rank d = n + it->second;
            adj[static_cast<std::size_t>(s)].push_back(d);
            adj[static_cast<std::size_t>(d)].push_back(s);
            ++edges;
        }
    }

    std::vector<Rank> cid(static_cast<std::size_t>(2 * n), Rank(-1));
    std::vector<Rank> source_count;
    std::vector<Rank> destination_count;
    std::vector<Rank> edge_count;
    for (Rank root = 0; root < 2 * n; ++root) {
        if (cid[static_cast<std::size_t>(root)] != Rank(-1)) continue;
        const Rank id = source_count.size();
        source_count.push_back(0);
        destination_count.push_back(0);
        edge_count.push_back(0);
        std::deque<Rank> q;
        cid[static_cast<std::size_t>(root)] = id;
        q.push_back(root);
        Rank degree_sum = 0;
        while (!q.empty()) {
            const Rank x = q.front();
            q.pop_front();
            if (x < n) ++source_count[static_cast<std::size_t>(id)];
            else ++destination_count[static_cast<std::size_t>(id)];
            degree_sum += adj[static_cast<std::size_t>(x)].size();
            for (Rank y : adj[static_cast<std::size_t>(x)]) {
                if (cid[static_cast<std::size_t>(y)] == Rank(-1)) {
                    cid[static_cast<std::size_t>(y)] = id;
                    q.push_back(y);
                }
            }
        }
        edge_count[static_cast<std::size_t>(id)] = degree_sum / 2;
    }

    ComponentStats st;
    st.components = source_count.size();
    st.edges = edges;
    for (Rank id = 0; id < st.components; ++id) {
        const Rank s = source_count[static_cast<std::size_t>(id)];
        const Rank d = destination_count[static_cast<std::size_t>(id)];
        const Rank e = edge_count[static_cast<std::size_t>(id)];
        if (s != d) fail("unbalanced production component W=" + std::to_string(W));
        st.max_pairs = std::max(st.max_pairs, s);
        st.max_edges = std::max(st.max_edges, e);
        st.max_cyclomatic = std::max(st.max_cyclomatic, e - (s + d) + 1);
    }

    const std::vector<Key> seeds = component_seeds(labels, W, p, reverse);
    if (seeds.size() != st.components) fail("seed/component count");
    std::vector<std::uint8_t> seed_seen(static_cast<std::size_t>(st.components));
    for (Key seed : seeds) {
        const auto it = src_rank.find(seed);
        if (it == src_rank.end()) fail("seed outside source layout");
        const Rank id = cid[static_cast<std::size_t>(it->second)];
        if (seed_seen[static_cast<std::size_t>(id)]++) fail("duplicate component seed");
    }
    for (std::uint8_t x : seed_seen)
        if (x != 1) fail("component without unique seed");

    return st;
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 10;
    if (maxW < 5 || maxW > 12) return 2;

    std::vector<std::vector<MateID>> words(static_cast<std::size_t>(maxW + 1));
    std::vector<Rank> M(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) {
        words[W] = gen_words(W);
        M[W] = words[W].size();
    }

    for (int W = 5; W <= maxW; ++W) {
        const Rank expect_components = M[W - 1] - M[W - 3];
        ComponentStats worst;
        for (bool reverse : {false, true}) {
            if (!reverse) {
                for (int p = W - 1; p >= 3; --p) {
                    const auto st = verify_components(words[W], words[W - 1], words[W - 1],
                                                      W, p, false);
                    if (st.components != expect_components) fail("forward component formula");
                    worst.components = st.components;
                    worst.edges = st.edges;
                    worst.max_pairs = std::max(worst.max_pairs, st.max_pairs);
                    worst.max_edges = std::max(worst.max_edges, st.max_edges);
                    worst.max_cyclomatic = std::max(worst.max_cyclomatic, st.max_cyclomatic);
                }
            } else {
                for (int p = 1; p <= W - 3; ++p) {
                    const auto st = verify_components(words[W], words[W - 1], words[W - 1],
                                                      W, p, true);
                    if (st.components != expect_components) fail("reverse component formula");
                    worst.components = st.components;
                    if (worst.edges && st.edges != worst.edges) fail("direction edge mismatch");
                    worst.edges = st.edges;
                    worst.max_pairs = std::max(worst.max_pairs, st.max_pairs);
                    worst.max_edges = std::max(worst.max_edges, st.max_edges);
                    worst.max_cyclomatic = std::max(worst.max_cyclomatic, st.max_cyclomatic);
                }
            }
        }

        const Rank pair_bound = Rank(W / 2 + 4);
        const Rank edge_bound = Rank(3 * (W / 2) + 5);
        if (worst.max_pairs > pair_bound || worst.max_edges > edge_bound)
            fail("observed component bound W=" + std::to_string(W));

        const Rank dim = M[W] + M[W - 1] - M[W - 2];
        std::cout << "W=" << W
                  << " states=" << dim
                  << " components=" << worst.components
                  << " expected_components=" << expect_components
                  << " avg_pairs=" << double(dim) / double(worst.components)
                  << " max_pairs=" << worst.max_pairs
                  << " max_edges=" << worst.max_edges
                  << " max_cyclomatic=" << worst.max_cyclomatic
                  << " unique_local_seed=1"
                  << " forward=OK reverse=OK\n";
    }

    const Rank m28 = 385719506620ULL;
    const Rank m27 = 135015505407ULL;
    const Rank m26 = 47337954326ULL;
    const Rank m25 = 16626415975ULL;
    const Rank dim28 = m28 + m27 - m26;
    const Rank comp28 = m27 - m25;
    std::cout << "W=28_theory states=" << dim28
              << " components=" << comp28
              << " avg_pairs=" << double(dim28) / double(comp28)
              << " observed_pair_bound=18"
              << " observed_edge_bound=47"
              << " component_table_bytes=0_candidate=1\n";
    std::cout << "ALL_OK production_component_partition=1\n";
    return 0;
}
