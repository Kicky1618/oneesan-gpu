#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_inverse_probe_main_unused
#include "gridfp_reduced_production_inverse_probe.cpp"
#pragma pop_macro("main")

#include <deque>
#include <limits>
#include <set>

namespace {

std::vector<Key> signed_shear_component_seeds(
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

struct Edge {
    int source = -1;
    int dest = -1;
    Coef coef = 0;
};

struct SignedShearStats {
    Rank components = 0;
    Rank states = 0;
    Rank edges = 0;
    Rank correction_edges = 0;
    Rank negative_matching = 0;
    Rank max_pairs = 0;
    Rank max_correction_in = 0;
    Rank max_correction_out = 0;
};

SignedShearStats verify_signed_shear_position(
    const std::vector<MateID>& main,
    const std::vector<MateID>& block,
    int W,
    int p,
    bool reverse
) {
    const int next = reverse ? p + 1 : p - 1;
    const std::vector<Key> src = layout(main, block, p);
    const std::vector<Key> dst = layout(main, block, next);
    if (src.size() != dst.size()) fail("signed shear non-square layout");
    const Rank n = src.size();

    std::map<Key, Rank> src_rank, dst_rank;
    for (Rank r = 0; r < n; ++r) {
        src_rank.emplace(src[static_cast<std::size_t>(r)], r);
        dst_rank.emplace(dst[static_cast<std::size_t>(r)], r);
    }

    const Rank bad = std::numeric_limits<Rank>::max();
    std::vector<std::uint8_t> source_seen(static_cast<std::size_t>(n));
    std::vector<std::uint8_t> dest_seen(static_cast<std::size_t>(n));
    SignedShearStats st;
    st.states = n;

    const auto seeds = signed_shear_component_seeds(block, W, p, reverse);
    st.components = seeds.size();

    for (Key seed : seeds) {
        std::set<Key> source_keys;
        std::set<Key> dest_keys;
        std::deque<Key> queue;
        source_keys.insert(seed);
        queue.push_back(seed);
        while (!queue.empty()) {
            const Key s = queue.front();
            queue.pop_front();
            for (const auto& [d, c] : reduced_step_basis(s, W, p, reverse)) {
                if (c != 1 && c != -1) fail("signed shear edge coefficient");
                if (!dest_keys.insert(d).second) continue;
                for (const auto& [pre, a] : inverse_reduced(d, W, p, reverse)) {
                    if (a != 1 && a != -1) fail("signed shear inverse coefficient");
                    if (source_keys.insert(pre).second) queue.push_back(pre);
                }
            }
        }
        if (source_keys.size() != dest_keys.size()) fail("signed shear unbalanced component");
        const int k = static_cast<int>(source_keys.size());
        st.max_pairs = std::max<Rank>(st.max_pairs, k);

        std::vector<Key> sources(source_keys.begin(), source_keys.end());
        std::vector<Key> dests(dest_keys.begin(), dest_keys.end());
        std::map<Key, int> slocal, dlocal;
        for (int q = 0; q < k; ++q) {
            slocal.emplace(sources[static_cast<std::size_t>(q)], q);
            dlocal.emplace(dests[static_cast<std::size_t>(q)], q);
        }

        std::vector<Edge> edges;
        std::vector<std::vector<int>> sout(static_cast<std::size_t>(k));
        std::vector<std::vector<int>> din(static_cast<std::size_t>(k));
        for (int s = 0; s < k; ++s) {
            const Rank sr = src_rank.at(sources[static_cast<std::size_t>(s)]);
            if (source_seen[static_cast<std::size_t>(sr)]++) fail("signed shear source overlap");
            for (const auto& [dkey, c] : reduced_step_basis(sources[static_cast<std::size_t>(s)], W, p, reverse)) {
                const int d = dlocal.at(dkey);
                const int ei = static_cast<int>(edges.size());
                edges.push_back(Edge{s, d, c});
                sout[static_cast<std::size_t>(s)].push_back(ei);
                din[static_cast<std::size_t>(d)].push_back(ei);
            }
        }
        for (int d = 0; d < k; ++d) {
            const Rank dr = dst_rank.at(dests[static_cast<std::size_t>(d)]);
            if (dest_seen[static_cast<std::size_t>(dr)]++) fail("signed shear destination overlap");
        }
        st.edges += edges.size();

        // A graph may contain undirected cycles, but repeated paired leaf
        // elimination still removes every source and destination. Therefore
        // the perfect matching is forced and unique without a matching search.
        std::vector<int> sdeg(static_cast<std::size_t>(k));
        std::vector<int> ddeg(static_cast<std::size_t>(k));
        std::vector<std::uint8_t> live_s(static_cast<std::size_t>(k), 1);
        std::vector<std::uint8_t> live_d(static_cast<std::size_t>(k), 1);
        std::deque<std::pair<bool, int>> leaves; // true = destination
        for (int q = 0; q < k; ++q) {
            sdeg[static_cast<std::size_t>(q)] = static_cast<int>(sout[static_cast<std::size_t>(q)].size());
            ddeg[static_cast<std::size_t>(q)] = static_cast<int>(din[static_cast<std::size_t>(q)].size());
            if (sdeg[static_cast<std::size_t>(q)] == 1) leaves.push_back({false, q});
            if (ddeg[static_cast<std::size_t>(q)] == 1) leaves.push_back({true, q});
        }

        std::vector<int> match_d(static_cast<std::size_t>(k), -1);
        std::vector<int> match_s(static_cast<std::size_t>(k), -1);
        std::vector<Coef> match_coef(static_cast<std::size_t>(k), 0);
        int matched = 0;
        while (!leaves.empty()) {
            const auto [right, v] = leaves.front();
            leaves.pop_front();
            int s = -1, d = -1, edge_id = -1;
            if (!right) {
                if (!live_s[static_cast<std::size_t>(v)] || sdeg[static_cast<std::size_t>(v)] != 1) continue;
                s = v;
                for (int ei : sout[static_cast<std::size_t>(s)]) {
                    const int z = edges[static_cast<std::size_t>(ei)].dest;
                    if (live_d[static_cast<std::size_t>(z)]) { d = z; edge_id = ei; break; }
                }
            } else {
                if (!live_d[static_cast<std::size_t>(v)] || ddeg[static_cast<std::size_t>(v)] != 1) continue;
                d = v;
                for (int ei : din[static_cast<std::size_t>(d)]) {
                    const int z = edges[static_cast<std::size_t>(ei)].source;
                    if (live_s[static_cast<std::size_t>(z)]) { s = z; edge_id = ei; break; }
                }
            }
            if (s < 0 || d < 0 || edge_id < 0) continue;
            if (match_d[static_cast<std::size_t>(s)] >= 0 || match_s[static_cast<std::size_t>(d)] >= 0)
                fail("signed shear duplicate match");
            const Coef a = edges[static_cast<std::size_t>(edge_id)].coef;
            if (a != 1 && a != -1) fail("signed shear nonunit pivot");
            match_d[static_cast<std::size_t>(s)] = d;
            match_s[static_cast<std::size_t>(d)] = s;
            match_coef[static_cast<std::size_t>(s)] = a;
            st.negative_matching += a < 0;
            ++matched;
            live_s[static_cast<std::size_t>(s)] = 0;
            live_d[static_cast<std::size_t>(d)] = 0;

            for (int ei : sout[static_cast<std::size_t>(s)]) {
                const int z = edges[static_cast<std::size_t>(ei)].dest;
                if (!live_d[static_cast<std::size_t>(z)]) continue;
                if (--ddeg[static_cast<std::size_t>(z)] == 1) leaves.push_back({true, z});
            }
            for (int ei : din[static_cast<std::size_t>(d)]) {
                const int z = edges[static_cast<std::size_t>(ei)].source;
                if (!live_s[static_cast<std::size_t>(z)]) continue;
                if (--sdeg[static_cast<std::size_t>(z)] == 1) leaves.push_back({false, z});
            }
        }
        if (matched != k) fail("signed shear leaf peeling did not fully match component");

        // Rename each matched destination into its source slot and absorb the
        // matching sign into that destination basis vector. The diagonal then
        // becomes +1. Every nonmatching edge is a signed correction u -> v.
        std::vector<std::vector<std::pair<int, Coef>>> incoming(static_cast<std::size_t>(k));
        std::vector<std::vector<int>> precedence(static_cast<std::size_t>(k));
        std::vector<int> indegree(static_cast<std::size_t>(k));
        for (const Edge& e : edges) {
            const int target = match_s[static_cast<std::size_t>(e.dest)];
            if (target < 0) fail("signed shear destination without owner");
            if (target == e.source) {
                if (e.coef != match_coef[static_cast<std::size_t>(target)])
                    fail("signed shear matching coefficient mismatch");
                continue;
            }
            const Coef c = match_coef[static_cast<std::size_t>(target)] * e.coef;
            incoming[static_cast<std::size_t>(target)].push_back({e.source, c});
            // target must be updated before source is itself overwritten.
            precedence[static_cast<std::size_t>(target)].push_back(e.source);
            ++indegree[static_cast<std::size_t>(e.source)];
            ++st.correction_edges;
        }
        for (int q = 0; q < k; ++q) {
            st.max_correction_in = std::max<Rank>(
                st.max_correction_in, incoming[static_cast<std::size_t>(q)].size());
            Rank out = 0;
            for (const auto& v : incoming)
                for (const auto& [s, c] : v) { (void)c; out += s == q; }
            st.max_correction_out = std::max(st.max_correction_out, out);
        }

        std::deque<int> ready;
        for (int q = 0; q < k; ++q)
            if (!indegree[static_cast<std::size_t>(q)]) ready.push_back(q);
        std::vector<int> order;
        while (!ready.empty()) {
            const int v = ready.front();
            ready.pop_front();
            order.push_back(v);
            for (int z : precedence[static_cast<std::size_t>(v)]) {
                int& deg = indegree[static_cast<std::size_t>(z)];
                if (deg <= 0) fail("signed shear precedence underflow");
                if (--deg == 0) ready.push_back(z);
            }
        }
        if (static_cast<int>(order.size()) != k)
            fail("signed shear correction graph has directed cycle");

        std::vector<Coef> input(static_cast<std::size_t>(k));
        std::vector<Coef> direct(static_cast<std::size_t>(k));
        std::vector<Coef> work(static_cast<std::size_t>(k));
        for (int s = 0; s < k; ++s) {
            input[static_cast<std::size_t>(s)] = Coef(1 + ((s * 97 + W * 31 + p * 7) % 1009));
            work[static_cast<std::size_t>(s)] = input[static_cast<std::size_t>(s)];
            for (int ei : sout[static_cast<std::size_t>(s)]) {
                const Edge& e = edges[static_cast<std::size_t>(ei)];
                direct[static_cast<std::size_t>(e.dest)] += e.coef * input[static_cast<std::size_t>(s)];
            }
        }
        for (int target : order) {
            for (const auto& [source, c] : incoming[static_cast<std::size_t>(target)])
                work[static_cast<std::size_t>(target)] += c * work[static_cast<std::size_t>(source)];
        }
        for (int s = 0; s < k; ++s) {
            const int d = match_d[static_cast<std::size_t>(s)];
            const Coef normalized = match_coef[static_cast<std::size_t>(s)] * direct[static_cast<std::size_t>(d)];
            if (work[static_cast<std::size_t>(s)] != normalized)
                fail("signed shear in-place arithmetic mismatch");
        }
    }

    for (Rank r = 0; r < n; ++r) {
        if (source_seen[static_cast<std::size_t>(r)] != 1 ||
            dest_seen[static_cast<std::size_t>(r)] != 1)
            fail("signed shear component coverage");
    }
    if (st.correction_edges + st.states != st.edges)
        fail("signed shear edge partition");
    return st;
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 10;
    if (maxW < 5 || maxW > 12) return 2;

    std::vector<std::vector<MateID>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 5; W <= maxW; ++W) {
        SignedShearStats worst{};
        bool first = true;
        for (int p = W - 1; p >= 3; --p) {
            const auto s = verify_signed_shear_position(words[W], words[W - 1], W, p, false);
            if (first) { worst = s; first = false; }
            else {
                if (s.states != worst.states || s.components != worst.components ||
                    s.edges != worst.edges || s.correction_edges != worst.correction_edges)
                    fail("signed shear forward position totals");
                worst.negative_matching = std::max(worst.negative_matching, s.negative_matching);
                worst.max_pairs = std::max(worst.max_pairs, s.max_pairs);
                worst.max_correction_in = std::max(worst.max_correction_in, s.max_correction_in);
                worst.max_correction_out = std::max(worst.max_correction_out, s.max_correction_out);
            }
        }
        for (int p = 1; p <= W - 3; ++p) {
            const auto s = verify_signed_shear_position(words[W], words[W - 1], W, p, true);
            if (s.states != worst.states || s.components != worst.components ||
                s.edges != worst.edges || s.correction_edges != worst.correction_edges)
                fail("signed shear reverse totals");
            worst.negative_matching = std::max(worst.negative_matching, s.negative_matching);
            worst.max_pairs = std::max(worst.max_pairs, s.max_pairs);
            worst.max_correction_in = std::max(worst.max_correction_in, s.max_correction_in);
            worst.max_correction_out = std::max(worst.max_correction_out, s.max_correction_out);
        }

        std::cout << "W=" << W
                  << " states=" << worst.states
                  << " components=" << worst.components
                  << " edges=" << worst.edges
                  << " correction_edges=" << worst.correction_edges
                  << " max_pairs=" << worst.max_pairs
                  << " negative_matching_max=" << worst.negative_matching
                  << " max_correction_in=" << worst.max_correction_in
                  << " max_correction_out=" << worst.max_correction_out
                  << " forced_matching=1 signed_basis=1 correction_DAG=1 inplace_exact=1\n";
    }

    constexpr Rank states28 = 473397057701ULL;
    constexpr Rank components28 = 118389089432ULL;
    const double one_buffer_gib = double(states28) * 4.0 / double(1ULL << 30);
    std::cout << "W=28_theory states=" << states28
              << " components=" << components28
              << " avg_pairs=" << double(states28) / double(components28)
              << " one_u32_buffer_GiB=" << one_buffer_gib
              << " per_8gpu_GiB=" << one_buffer_gib / 8.0
              << " double_buffer_required=0_if_inplace_layout_resolved"
              << '\n';
    std::cout << "ALL_OK production_signed_shear=1\n";
    return 0;
}
