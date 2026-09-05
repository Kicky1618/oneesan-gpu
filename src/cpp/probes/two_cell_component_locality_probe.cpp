#pragma push_macro("main")
#undef main
#define main two_cell_factorized_gather_probe_main_unused
#include "two_cell_factorized_gather_probe.cpp"
#pragma pop_macro("main")

#include <deque>

namespace {

std::set<Key> locality_component_sources(const Key& seed, int W, int i) {
    std::set<Key> sources{seed};
    std::deque<Key> queue{seed};
    while (!queue.empty()) {
        const Key src = queue.front();
        queue.pop_front();
        for (const auto& [dst, c] : K_basis(src, W, i)) {
            if (c != 1) fail("locality nonunit");
            for (const Key& pre : inverse_K(dst, W, i))
                if (sources.insert(pre).second) queue.push_back(pre);
        }
    }
    return sources;
}

Rank ceil_div(Rank x, Rank y) { return (x + y - 1) / y; }

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 12;
    if (maxW < 5 || maxW > 15) return 2;

    std::vector<std::vector<Word>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 5; W <= maxW; ++W) {
        FactorTables tables(W);
        long double total_values = 0;
        long double actual_32b = 0, ideal_32b = 0;
        long double actual_128b = 0, ideal_128b = 0;
        long double span_sum = 0;
        Rank components = 0;
        Rank max_span = 0;
        Rank max_pairs = 0;

        for (int i = 0; i <= W - 4; ++i) {
            FactorizedCodec codec(tables, i);
            for (const Word& u : words[W - 2]) {
                const Key seed = project_key(Key{'C', u}, i, W);
                const auto sources = locality_component_sources(seed, W, i);
                std::vector<Rank> rank;
                rank.reserve(sources.size());
                for (const Key& s : sources) rank.push_back(codec.rank(s));
                std::sort(rank.begin(), rank.end());

                std::set<Rank> sector32;
                std::set<Rank> sector128;
                for (Rank r : rank) {
                    sector32.insert(r / 8);   // eight uint32 values
                    sector128.insert(r / 32); // thirty-two uint32 values
                }

                const Rank pairs = rank.size();
                const Rank span = rank.empty() ? 0 : rank.back() - rank.front();
                total_values += pairs;
                actual_32b += sector32.size();
                ideal_32b += ceil_div(pairs, 8);
                actual_128b += sector128.size();
                ideal_128b += ceil_div(pairs, 32);
                span_sum += span;
                max_span = std::max(max_span, span);
                max_pairs = std::max(max_pairs, pairs);
                ++components;
            }
        }

        std::cout << "W=" << W
                  << " components_checked=" << components
                  << " avg_pairs=" << double(total_values / components)
                  << " max_pairs=" << max_pairs
                  << " factorized_32B_sectors_per_component=" << double(actual_32b / components)
                  << " component_major_32B_sectors_per_component=" << double(ideal_32b / components)
                  << " factorized_128B_sectors_per_component=" << double(actual_128b / components)
                  << " component_major_128B_sectors_per_component=" << double(ideal_128b / components)
                  << " sector128_ratio=" << double(actual_128b / ideal_128b)
                  << " avg_factorized_rank_span=" << double(span_sum / components)
                  << " max_factorized_rank_span=" << max_span
                  << " OK\n";
    }

    std::cout << "ALL_OK component_major_locality_candidate=1\n";
    return 0;
}
