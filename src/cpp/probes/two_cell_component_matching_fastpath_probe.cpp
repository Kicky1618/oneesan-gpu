#pragma push_macro("main")
#undef main
#define main two_cell_component_matching_probe_main_unused
#include "two_cell_component_matching_probe.cpp"
#pragma pop_macro("main")

namespace {

std::uint32_t adjacency_mask(
    const std::vector<Key>& src,
    int s,
    int W,
    int i
) {
    std::uint32_t mask = 0;
    for (const auto& [d, c] : K_basis(src[static_cast<std::size_t>(s)], W, i)) {
        if (c != 1) fail("fastpath coefficient");
        int t = -1;
        for (int q = 0; q < static_cast<int>(src.size()); ++q) {
            if (recouple_coordinate(src[static_cast<std::size_t>(q)], i) == d) {
                t = q;
                break;
            }
        }
        if (t < 0) fail("fastpath destination coordinate");
        mask |= std::uint32_t(1) << t;
    }
    return mask;
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 14;
    if (maxW < 5 || maxW > 15) return 2;

    std::vector<std::vector<Word>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 5; W <= maxW; ++W) {
        Rank singleton = 0;
        Rank triple = 0;
        Rank generic = 0;
        for (int i = 0; i <= W - 4; ++i) {
            for (const Word& u : words[W - 2]) {
                const auto packed = packed_direct_component_sources(pack_word(u), W, i);
                std::vector<Key> src;
                for (int q = 0; q < packed.size; ++q)
                    src.push_back(unpack_key(packed.value[q]));

                if (src.size() == 1) {
                    if (adjacency_mask(src, 0, W, i) != 0x1u)
                        fail("singleton fastpath adjacency");
                    ++singleton;
                    continue;
                }
                if (src.size() == 3) {
                    const std::uint32_t want[3] = {0x4u, 0x2u, 0x7u};
                    for (int s = 0; s < 3; ++s)
                        if (adjacency_mask(src, s, W, i) != want[s])
                            fail("triple fastpath adjacency W=" + std::to_string(W));

                    const std::uint32_t mod = 1000000007u;
                    std::uint32_t input[3] = {
                        static_cast<std::uint32_t>(17 + i),
                        static_cast<std::uint32_t>(31 + W),
                        static_cast<std::uint32_t>(47 + i + W)
                    };
                    std::uint32_t expected[3]{};
                    for (int s = 0; s < 3; ++s) {
                        const std::uint32_t m = want[s];
                        for (int t = 0; t < 3; ++t) if ((m >> t) & 1u)
                            expected[t] = static_cast<std::uint32_t>(
                                (static_cast<std::uint64_t>(expected[t]) + input[s]) % mod);
                    }
                    std::uint32_t got[3] = {
                        input[2],
                        static_cast<std::uint32_t>((static_cast<std::uint64_t>(input[1]) + input[2]) % mod),
                        static_cast<std::uint32_t>((static_cast<std::uint64_t>(input[0]) + input[2]) % mod)
                    };
                    for (int t = 0; t < 3; ++t)
                        if (got[t] != expected[t]) fail("triple fastpath arithmetic");
                    ++triple;
                    continue;
                }
                ++generic;
            }
        }

        const Rank positions = W - 3;
        const Rank m2 = words[W - 2].size();
        const Rank m3 = words[W - 3].size();
        if (singleton != positions * m3 ||
            triple != positions * (m2 - 2 * m3) ||
            generic != positions * m3)
            fail("matching fastpath class counts W=" + std::to_string(W));

        std::cout << "W=" << W
                  << " singleton=" << singleton
                  << " triple=" << triple
                  << " deep=" << generic
                  << " singleton_adjacency=1"
                  << " triple_adjacency=[4,2,7]"
                  << " triple_matching=[2,1,0]"
                  << " triple_residual_adds=2"
                  << " OK\n";
    }

    const std::uint64_t M26 = 47337954326ULL;
    const std::uint64_t M25 = 16626415975ULL;
    const std::uint64_t R28 = 165727043758ULL;
    const std::uint64_t fast_components = M26 - M25;
    const std::uint64_t fast_states = M25 + 3ULL * (M26 - 2ULL * M25);
    std::cout << "W=28_theory fast_components=" << fast_components
              << " component_fraction=" << double(fast_components) / double(M26)
              << " fast_states=" << fast_states
              << " state_fraction=" << double(fast_states) / double(R28)
              << " K_step_calls_fastpath=0 leaf_peeling_fastpath=0\n";
    std::cout << "ALL_OK component_matching_fastpaths=1\n";
    return 0;
}
