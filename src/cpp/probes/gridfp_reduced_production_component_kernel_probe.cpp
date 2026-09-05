#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_inverse_probe_main_unused
#include "gridfp_reduced_production_inverse_probe.cpp"
#pragma pop_macro("main")

#include <deque>
#include <set>

namespace {

std::vector<Key> local_component_seeds(
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

struct LocalKernelStats {
    Rank components = 0;
    Rank states = 0;
    Rank edges = 0;
    Rank max_pairs = 0;
    Rank max_edges = 0;
};

LocalKernelStats verify_local_kernel(
    const std::vector<MateID>& main,
    const std::vector<MateID>& block,
    int W,
    int p,
    bool reverse
) {
    const int next = reverse ? p + 1 : p - 1;
    const std::vector<Key> src = layout(main, block, p);
    const std::vector<Key> dst = layout(main, block, next);
    const Rank n = src.size();
    if (dst.size() != n) fail("local kernel square layout");

    std::map<Key, Rank> src_rank, dst_rank;
    for (Rank r = 0; r < n; ++r) {
        src_rank.emplace(src[static_cast<std::size_t>(r)], r);
        dst_rank.emplace(dst[static_cast<std::size_t>(r)], r);
    }

    std::vector<Coef> reference(static_cast<std::size_t>(n));
    std::vector<Coef> output(static_cast<std::size_t>(n));
    for (Rank s = 0; s < n; ++s) {
        const Coef value = Coef(1 + (s % 1000003));
        for (const auto& [d, c] : reduced_step_basis(src[static_cast<std::size_t>(s)], W, p, reverse))
            reference[static_cast<std::size_t>(dst_rank.at(d))] += c * value;
    }

    std::vector<std::uint8_t> source_seen(static_cast<std::size_t>(n));
    std::vector<std::uint8_t> destination_seen(static_cast<std::size_t>(n));
    const std::vector<Key> seeds = local_component_seeds(block, W, p, reverse);

    LocalKernelStats st;
    st.components = seeds.size();
    st.states = n;

    for (Key seed : seeds) {
        if (!src_rank.count(seed)) fail("local kernel seed outside source layout");
        std::set<Key> sources;
        std::set<Key> destinations;
        std::deque<Key> queue;
        sources.insert(seed);
        queue.push_back(seed);

        while (!queue.empty()) {
            const Key s = queue.front();
            queue.pop_front();
            for (const auto& [d, c] : reduced_step_basis(s, W, p, reverse)) {
                if (c != 1 && c != -1) fail("local kernel edge coefficient");
                if (!destinations.insert(d).second) continue;
                for (const auto& [pre, a] : inverse_reduced(d, W, p, reverse)) {
                    if (a != 1 && a != -1) fail("local kernel inverse coefficient");
                    if (sources.insert(pre).second) queue.push_back(pre);
                }
            }
        }

        if (sources.size() != destinations.size()) fail("local kernel unbalanced component");
        const Rank pairs = sources.size();
        st.max_pairs = std::max(st.max_pairs, pairs);

        Rank component_edges = 0;
        for (Key s : sources) {
            const auto sit = src_rank.find(s);
            if (sit == src_rank.end()) fail("local source outside layout");
            const Rank sr = sit->second;
            if (source_seen[static_cast<std::size_t>(sr)]++) fail("local source overlap");
            const Coef value = Coef(1 + (sr % 1000003));
            for (const auto& [d, c] : reduced_step_basis(s, W, p, reverse)) {
                if (!destinations.count(d)) fail("local edge escapes component");
                const auto dit = dst_rank.find(d);
                if (dit == dst_rank.end()) fail("local destination outside layout");
                output[static_cast<std::size_t>(dit->second)] += c * value;
                ++component_edges;
            }
        }
        for (Key d : destinations) {
            const Rank dr = dst_rank.at(d);
            if (destination_seen[static_cast<std::size_t>(dr)]++) fail("local destination overlap");
        }
        st.edges += component_edges;
        st.max_edges = std::max(st.max_edges, component_edges);
    }

    for (Rank r = 0; r < n; ++r) {
        if (source_seen[static_cast<std::size_t>(r)] != 1 ||
            destination_seen[static_cast<std::size_t>(r)] != 1)
            fail("local component coverage W=" + std::to_string(W));
        if (output[static_cast<std::size_t>(r)] != reference[static_cast<std::size_t>(r)])
            fail("local component arithmetic W=" + std::to_string(W));
    }
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
        LocalKernelStats worst;
        for (int p = W - 1; p >= 3; --p) {
            const auto st = verify_local_kernel(words[W], words[W - 1], W, p, false);
            if (st.components != expect_components) fail("local forward component count");
            if (!worst.states) worst = st;
            if (st.states != worst.states || st.components != worst.components || st.edges != worst.edges)
                fail("local forward position totals");
            worst.max_pairs = std::max(worst.max_pairs, st.max_pairs);
            worst.max_edges = std::max(worst.max_edges, st.max_edges);
        }
        for (int p = 1; p <= W - 3; ++p) {
            const auto st = verify_local_kernel(words[W], words[W - 1], W, p, true);
            if (st.components != expect_components || st.states != worst.states || st.edges != worst.edges)
                fail("local reverse totals");
            worst.max_pairs = std::max(worst.max_pairs, st.max_pairs);
            worst.max_edges = std::max(worst.max_edges, st.max_edges);
        }

        if (worst.max_pairs > Rank(W / 2 + 4)) fail("local pair bound");
        if (worst.max_edges > Rank(3 * (W / 2) + 5)) fail("local edge bound");
        std::cout << "W=" << W
                  << " states=" << worst.states
                  << " components=" << worst.components
                  << " avg_pairs=" << double(worst.states) / double(worst.components)
                  << " max_pairs=" << worst.max_pairs
                  << " edges=" << worst.edges
                  << " max_edges=" << worst.max_edges
                  << " source_load_once=1 destination_store_once=1"
                  << " component_table_bytes=0 inverse_table_bytes=0"
                  << " arithmetic=OK forward=OK reverse=OK\n";
    }

    const Rank m28 = 385719506620ULL;
    const Rank m27 = 135015505407ULL;
    const Rank m26 = 47337954326ULL;
    const Rank m25 = 16626415975ULL;
    const Rank dim28 = m28 + m27 - m26;
    const Rank comp28 = m27 - m25;
    const double gib_per_stream = double(dim28) * 4.0 / double(1ULL << 30);
    std::cout << "W=28_theory states=" << dim28
              << " components=" << comp28
              << " avg_pairs=" << double(dim28) / double(comp28)
              << " pair_bound_candidate=18 edge_bound_candidate=47"
              << " u32_stream_GiB=" << gib_per_stream
              << " ideal_count_traffic_GiB_per_step=" << 2.0 * gib_per_stream
              << "\n";
    std::cout << "ALL_OK production_component_local_kernel=1\n";
    return 0;
}
