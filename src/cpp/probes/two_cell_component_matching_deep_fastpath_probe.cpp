#pragma push_macro("main")
#undef main
#define main two_cell_component_matching_fastpath_probe_main_unused
#include "two_cell_component_matching_fastpath_probe.cpp"
#pragma pop_macro("main")

namespace {

std::uint32_t deep_expected_mask(const Word& u, int i, int n, int s) {
    const std::string local = u.substr(static_cast<std::size_t>(i), 2);
    if (local == "LR") {
        if (s == 0) return 1u << 2;
        if (s == 1) return 1u << 1;
        if (s == 2) return 0x7u;
        if (s == 3) return (1u << 1) | (1u << 3);
        return (1u << 3) | (1u << s);
    }
    if (local == "RN") {
        if (s == 0) return 1u << 2;
        if (s == 1) return 1u << 1;
        if (s == 2) return (1u << 1) | (1u << 2) | (1u << (n - 1));
        if (s == 3) return (1u << 0) | (1u << 3);
        return (1u << 3) | (1u << s);
    }
    return 0;
}

int deep_expected_match(const Word& u, int i, int n, int s) {
    const std::string local = u.substr(static_cast<std::size_t>(i), 2);
    if (local == "LR") {
        if (s == 0) return 2;
        if (s == 1) return 1;
        if (s == 2) return 0;
        return s;
    }
    if (local == "RN") {
        if (s == 0) return 2;
        if (s == 1) return 1;
        if (s == 2) return n - 1;
        if (s == 3) return 0;
        if (s == n - 1) return 3;
        return s;
    }
    return -1;
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 14;
    if (maxW < 5 || maxW > 15) return 2;

    std::vector<std::vector<Word>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 5; W <= maxW; ++W) {
        Rank rn = 0, lr = 0, ln = 0;
        Rank checked_edges = 0;
        Rank max_fast_size = 0;
        for (int i = 0; i <= W - 4; ++i) {
            for (const Word& u : words[W - 2]) {
                const std::string local = u.substr(static_cast<std::size_t>(i), 2);
                if (local != "RN" && local != "LN" && local != "LR") continue;

                const auto packed = packed_direct_component_sources(pack_word(u), W, i);
                std::vector<Key> src;
                for (int q = 0; q < packed.size; ++q)
                    src.push_back(unpack_key(packed.value[q]));
                const int n = static_cast<int>(src.size());
                if (n < 5) fail("deep fastpath size");

                if (local == "LN") {
                    ++ln;
                    continue;
                }
                if (local == "RN") ++rn; else ++lr;
                max_fast_size = std::max<Rank>(max_fast_size, n);

                std::vector<int> seen_dst(static_cast<std::size_t>(n), 0);
                int residual = 0;
                for (int s = 0; s < n; ++s) {
                    const std::uint32_t got = adjacency_mask(src, s, W, i);
                    const std::uint32_t want = deep_expected_mask(u, i, n, s);
                    if (got != want)
                        fail("deep fastpath adjacency W=" + std::to_string(W) +
                             " i=" + std::to_string(i) + " local=" + local);
                    const int t = deep_expected_match(u, i, n, s);
                    if (t < 0 || t >= n || !((got >> t) & 1u))
                        fail("deep fastpath matching edge");
                    if (++seen_dst[static_cast<std::size_t>(t)] != 1)
                        fail("deep fastpath matching collision");
                    residual += oneesan::twocell::popcount32(got) - 1;
                    checked_edges += oneesan::twocell::popcount32(got);
                }
                for (int t = 0; t < n; ++t)
                    if (seen_dst[static_cast<std::size_t>(t)] != 1)
                        fail("deep fastpath matching coverage");
                if (residual != n - 1) fail("deep fastpath residual count");
            }
        }

        std::cout << "W=" << W
                  << " RN=" << rn
                  << " LR=" << lr
                  << " LN_generic=" << ln
                  << " checked_edges=" << checked_edges
                  << " max_fast_size=" << max_fast_size
                  << " RN_formula=OK LR_formula=OK\n";
    }

    // W=28, label length 26, 25 interior positions.  Dynamic path counting
    // gives these exact local-pair totals across the whole forward sweep.
    const std::uint64_t positions = 25;
    const std::uint64_t M26 = 47337954326ULL;
    const std::uint64_t M25 = 16626415975ULL;
    const std::uint64_t shallow_fast = positions * (M26 - M25);
    const std::uint64_t rn_total = 143009973875ULL;
    const std::uint64_t lr_total = 146266867600ULL;
    const std::uint64_t ln_total = 126383557900ULL;
    const std::uint64_t all_components = positions * M26;
    const std::uint64_t closed = shallow_fast + rn_total + lr_total;
    if (rn_total + lr_total + ln_total != positions * M25)
        fail("W28 deep local pair totals");
    std::cout << "W=28_sweep_theory closed_components=" << closed
              << " all_components=" << all_components
              << " closed_fraction=" << double(closed) / double(all_components)
              << " deep_RN_LR_fraction="
              << double(rn_total + lr_total) / double(positions * M25)
              << " LN_generic_fraction=" << double(ln_total) / double(all_components)
              << "\n";
    std::cout << "ALL_OK RN_LR_deep_matching_fastpaths=1\n";
    return 0;
}
