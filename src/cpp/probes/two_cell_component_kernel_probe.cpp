#pragma push_macro("main")
#undef main
#define main two_cell_inverse_gather_probe_main_unused
#include "two_cell_inverse_gather_probe.cpp"
#pragma pop_macro("main")

#include <deque>

namespace {

struct ComponentKernelStats {
    Rank components = 0;
    Rank states = 0;
    Rank edges = 0;
    Rank max_pairs = 0;
    Rank global_value_loads = 0;
    Rank global_value_stores = 0;
    Rank local_adds = 0;
};

ComponentKernelStats verify_component_kernel(
    int W,
    int i,
    const std::vector<std::vector<Word>>& words
) {
    const ReducedRankCodec src_codec(W, i);
    const ReducedRankCodec dst_codec(W, i + 1);
    const WordRankCodec component_codec(W - 2);
    const Rank n = src_codec.size();
    if (dst_codec.size() != n || component_codec.size() != words[W - 2].size())
        fail("component codec dimensions W=" + std::to_string(W));

    std::vector<std::uint8_t> source_seen(static_cast<std::size_t>(n));
    std::vector<std::uint8_t> destination_seen(static_cast<std::size_t>(n));
    std::vector<std::uint64_t> output(static_cast<std::size_t>(n));
    std::vector<std::uint64_t> reference(static_cast<std::size_t>(n));
    for (Rank s = 0; s < n; ++s) {
        const Key src = src_codec.unrank(s);
        const std::uint64_t value = 1 + ((s * 0x9e3779b97f4a7c15ULL) ^
                                         (Rank(W) << 32) ^ Rank(i));
        for (const auto& [dst, c] : K_basis(src, W, i)) {
            if (c != 1) fail("component reference nonunit");
            reference[dst_codec.rank(dst)] += value;
        }
    }

    ComponentKernelStats st;
    st.components = component_codec.size();
    st.states = n;

    for (Rank component_rank = 0; component_rank < component_codec.size(); ++component_rank) {
        const Word label = component_codec.unrank(component_rank);
        const Key seed = project_key(Key{'C', label}, i, W);
        if (src_codec.rank(seed) >= n) fail("component seed rank");

        std::set<Key> sources;
        std::set<Key> destinations;
        std::deque<Key> source_queue;
        sources.insert(seed);
        source_queue.push_back(seed);

        while (!source_queue.empty()) {
            const Key src = source_queue.front();
            source_queue.pop_front();
            for (const auto& [dst, c] : K_basis(src, W, i)) {
                if (c != 1) fail("component local nonunit");
                if (!destinations.insert(dst).second) continue;
                for (const Key& pre : inverse_K(dst, W, i)) {
                    if (sources.insert(pre).second) source_queue.push_back(pre);
                }
            }
        }

        if (sources.size() != destinations.size())
            fail("component local imbalance W=" + std::to_string(W) +
                 " i=" + std::to_string(i));
        const Rank pairs = sources.size();
        st.max_pairs = std::max(st.max_pairs, pairs);

        Rank component_edges = 0;
        for (const Key& src : sources) {
            const Rank sr = src_codec.rank(src);
            if (sr >= n || source_seen[sr])
                fail("component source overlap W=" + std::to_string(W));
            source_seen[sr] = 1;
            ++st.global_value_loads;

            const std::uint64_t value = 1 + ((sr * 0x9e3779b97f4a7c15ULL) ^
                                             (Rank(W) << 32) ^ Rank(i));
            for (const auto& [dst, c] : K_basis(src, W, i)) {
                if (c != 1 || !destinations.count(dst))
                    fail("component edge escapes local tree");
                output[dst_codec.rank(dst)] += value;
                ++component_edges;
            }
        }
        for (const Key& dst : destinations) {
            const Rank dr = dst_codec.rank(dst);
            if (dr >= n || destination_seen[dr])
                fail("component destination overlap W=" + std::to_string(W));
            destination_seen[dr] = 1;
            ++st.global_value_stores;
        }

        if (component_edges + 1 != 2 * pairs)
            fail("component is not a tree W=" + std::to_string(W) +
                 " i=" + std::to_string(i));
        st.edges += component_edges;
        st.local_adds += component_edges - pairs;
    }

    for (Rank r = 0; r < n; ++r) {
        if (!source_seen[r] || !destination_seen[r])
            fail("component coverage W=" + std::to_string(W));
        if (output[r] != reference[r])
            fail("component output mismatch W=" + std::to_string(W) +
                 " i=" + std::to_string(i) + " r=" + std::to_string(r));
    }

    const Rank expected_edges = 2 * n - component_codec.size();
    if (st.global_value_loads != n || st.global_value_stores != n ||
        st.edges != expected_edges || st.local_adds != st.edges - n)
        fail("component traffic formula W=" + std::to_string(W));
    return st;
}

Rank count_words_component(int W) {
    WordRankCodec codec(W);
    return codec.size();
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 12;
    if (maxW < 4 || maxW > 15) return 2;

    std::vector<std::vector<Word>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 4; W <= maxW; ++W) {
        ComponentKernelStats worst;
        for (int i = 0; i <= W - 4; ++i) {
            const ComponentKernelStats s = verify_component_kernel(W, i, words);
            if (!worst.states) worst = s;
            if (s.states != worst.states || s.components != worst.components ||
                s.edges != worst.edges || s.global_value_loads != worst.global_value_loads ||
                s.global_value_stores != worst.global_value_stores ||
                s.local_adds != worst.local_adds)
                fail("component position-dependent totals W=" + std::to_string(W));
            worst.max_pairs = std::max(worst.max_pairs, s.max_pairs);
        }
        std::cout << "W=" << W
                  << " states=" << worst.states
                  << " components=" << worst.components
                  << " avg_pairs=" << double(worst.states) / double(worst.components)
                  << " max_pairs=" << worst.max_pairs
                  << " edges=" << worst.edges
                  << " global_loads=" << worst.global_value_loads
                  << " global_stores=" << worst.global_value_stores
                  << " local_adds=" << worst.local_adds
                  << " component_table_bytes=0 csr_bytes=0"
                  << " local_reconstruction=1"
                  << " OK\n";
    }

    const Rank m27 = count_words_component(27);
    const Rank m26 = count_words_component(26);
    const Rank m25 = count_words_component(25);
    const Rank n28 = m27 + m26 - m25;
    const Rank edges28 = 2 * n28 - m26;
    const Rank saved_loads28 = edges28 - n28;
    const double saved_gib28 = double(saved_loads28) * 4.0 / double(1ull << 30);
    std::cout << "W=28_theory states=" << n28
              << " components=" << m26
              << " formula_gather_loads=" << edges28
              << " component_kernel_loads=" << n28
              << " saved_value_loads=" << saved_loads28
              << " saved_u32_gib_per_step=" << saved_gib28
              << " avg_pairs=" << double(n28) / double(m26)
              << "\n";
    std::cout << "ALL_OK component_local_kernel=1 global_value_load_once=1"
              << " global_value_store_once=1\n";
    return 0;
}
