#pragma push_macro("main")
#undef main
#define main two_cell_component_matching_deep_fastpath_probe_main_unused
#include "two_cell_component_matching_deep_fastpath_probe.cpp"
#pragma pop_macro("main")

namespace {

int ln_pivot_index(const std::vector<Key>& src, int W, int i) {
    if (src.size() < 5) fail("LN pivot component size");
    const std::uint32_t m = adjacency_mask(src, 2, W, i);
    const std::uint32_t extra = m & ~0x6u; // remove destinations 1 and 2
    if (!extra || (extra & (extra - 1))) fail("LN pivot not unique");
    const int k = oneesan::twocell::ctz32(extra);
    if (k < 4 || k >= static_cast<int>(src.size())) fail("LN pivot range");
    return k;
}

std::uint32_t ln_expected_mask(int n, int pivot, int s) {
    if (s == 0) return 1u << 2;
    if (s == 1) return 1u << 1;
    if (s == 2) return (1u << 1) | (1u << 2) | (1u << pivot);
    if (s == 3) return (1u << 0) | (1u << 3);
    return (1u << 3) | (1u << s);
}

int ln_expected_match(int pivot, int s) {
    if (s == 0) return 2;
    if (s == 1) return 1;
    if (s == 2) return pivot;
    if (s == 3) return 0;
    if (s == pivot) return 3;
    return s;
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 14;
    if (maxW < 6 || maxW > 15) return 2;

    std::vector<std::vector<Word>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 6; W <= maxW; ++W) {
        Rank checked = 0;
        Rank edge_checks = 0;
        Rank max_size = 0;
        Rank min_pivot = Rank(-1), max_pivot = 0;
        for (int i = 0; i <= W - 4; ++i) {
            for (const Word& u : words[W - 2]) {
                if (u.substr(static_cast<std::size_t>(i), 2) != "LN") continue;
                const auto packed = packed_direct_component_sources(pack_word(u), W, i);
                std::vector<Key> src;
                for (int q = 0; q < packed.size; ++q)
                    src.push_back(unpack_key(packed.value[q]));
                const int n = static_cast<int>(src.size());
                if (n < 5) fail("LN pivot deep size");
                const int pivot = ln_pivot_index(src, W, i);
                min_pivot = std::min<Rank>(min_pivot, pivot);
                max_pivot = std::max<Rank>(max_pivot, pivot);
                max_size = std::max<Rank>(max_size, n);

                std::vector<int> used(static_cast<std::size_t>(n));
                int residual = 0;
                for (int s = 0; s < n; ++s) {
                    const std::uint32_t got = adjacency_mask(src, s, W, i);
                    const std::uint32_t want = ln_expected_mask(n, pivot, s);
                    if (got != want)
                        fail("LN pivot adjacency W=" + std::to_string(W) +
                             " i=" + std::to_string(i));
                    const int t = ln_expected_match(pivot, s);
                    if (++used[static_cast<std::size_t>(t)] != 1)
                        fail("LN pivot matching collision");
                    if (!((got >> t) & 1u)) fail("LN pivot matching edge");
                    residual += oneesan::twocell::popcount32(got) - 1;
                    edge_checks += oneesan::twocell::popcount32(got);
                }
                for (int t = 0; t < n; ++t)
                    if (used[static_cast<std::size_t>(t)] != 1)
                        fail("LN pivot matching coverage");
                if (residual != n - 1) fail("LN pivot residual count");
                ++checked;
            }
        }
        std::cout << "W=" << W
                  << " LN_components=" << checked
                  << " edge_checks=" << edge_checks
                  << " max_size=" << max_size
                  << " min_pivot=" << min_pivot
                  << " max_pivot=" << max_pivot
                  << " K_step_calls_for_matching=1"
                  << " leaf_peeling=0"
                  << " formula=OK\n";
    }

    const std::uint64_t all = 25ULL * 47337954326ULL;
    const std::uint64_t ln = 126383557900ULL;
    std::cout << "W=28_sweep_theory matching_K_step_calls=" << ln
              << " components=" << all
              << " K_step_calls_per_component=" << double(ln) / double(all)
              << " matching_leaf_peeling_calls=0\n";
    std::cout << "ALL_OK LN_one_pivot_matching=1\n";
    return 0;
}
