#pragma push_macro("main")
#undef main
#define main two_cell_stationary_rank_probe_main_unused
#include "two_cell_stationary_rank_probe.cpp"
#pragma pop_macro("main")

namespace {

struct FusionDsu {
    std::vector<int> p;
    explicit FusionDsu(int n) : p(static_cast<std::size_t>(n), -1) {}
    int find(int x) {
        if (p[static_cast<std::size_t>(x)] < 0) return x;
        return p[static_cast<std::size_t>(x)] = find(p[static_cast<std::size_t>(x)]);
    }
    void unite(int a, int b) {
        a = find(a); b = find(b);
        if (a == b) return;
        if (p[static_cast<std::size_t>(a)] > p[static_cast<std::size_t>(b)]) std::swap(a, b);
        p[static_cast<std::size_t>(a)] += p[static_cast<std::size_t>(b)];
        p[static_cast<std::size_t>(b)] = a;
    }
    Rank size_of_root(int r) const { return static_cast<Rank>(-p[static_cast<std::size_t>(r)]); }
};

Rank primitive_count(int occupied, const oneesan::twocell::RankTables& t) {
    if (occupied <= 0 || occupied > oneesan::twocell::kMaxWidth || !(occupied & 1)) return 0;
    return t.primitive[occupied][1];
}

Rank fusion_block_size(
    int steps,
    int outer_ones,
    const oneesan::twocell::RankTables& t
) {
    Rank z = 0;
    // A coordinates: k+2 support bits may vary inside the fused window.
    for (int local = 0; local <= steps + 2; ++local)
        z += t.choose[steps + 2][local] * primitive_count(outer_ones + local, t);
    // C coordinates: one distinguished active support bit is fixed occupied;
    // the remaining k local support bits vary.
    for (int local = 0; local <= steps; ++local)
        z += t.choose[steps][local] * primitive_count(outer_ones + 1 + local, t);
    return z;
}

std::map<Rank, Rank> expected_fusion_size_histogram(
    int W,
    int steps,
    const oneesan::twocell::RankTables& t
) {
    const int outer_bits = W - steps - 3;
    std::map<Rank, Rank> out;
    for (int o = 0; o <= outer_bits; ++o)
        out[fusion_block_size(steps, o, t)] += t.choose[outer_bits][o];
    return out;
}

std::map<Rank, Rank> actual_fusion_size_histogram(
    int W,
    int start,
    int steps,
    const std::vector<std::vector<Word>>& words,
    const oneesan::twocell::RankTables& rt,
    const oneesan::twocell::StationaryRankTables& st
) {
    const int n = static_cast<int>(st.total[W]);
    FusionDsu dsu(n);
    for (int i = start; i < start + steps; ++i) {
        for (const Key& s : q_basis(W, i, words)) {
            const int sr = static_cast<int>(oneesan::twocell::stationary_rank(
                stationary_pack(s), W, i, rt, st));
            for (const auto& [d, c] : K_basis(s, W, i)) {
                if (c != 1) fail("fusion nonunit interior edge");
                const int dr = static_cast<int>(oneesan::twocell::stationary_rank(
                    stationary_pack(d), W, i + 1, rt, st));
                dsu.unite(sr, dr);
            }
        }
    }
    std::map<Rank, Rank> hist;
    for (int r = 0; r < n; ++r)
        if (dsu.find(r) == r) ++hist[dsu.size_of_root(r)];
    return hist;
}

struct CapacityStats {
    Rank fit_blocks = 0;
    Rank all_blocks = 0;
    Rank fit_states = 0;
    Rank all_states = 0;
    int max_outer_ones = -1;
};

CapacityStats capacity_stats(
    int W,
    int steps,
    Rank bytes,
    const oneesan::twocell::RankTables& t
) {
    CapacityStats s;
    const int outer_bits = W - steps - 3;
    for (int o = 0; o <= outer_bits; ++o) {
        const Rank blocks = t.choose[outer_bits][o];
        const Rank states = fusion_block_size(steps, o, t);
        s.all_blocks += blocks;
        s.all_states += blocks * states;
        if (states * sizeof(std::uint32_t) <= bytes) {
            s.fit_blocks += blocks;
            s.fit_states += blocks * states;
            s.max_outer_ones = o;
        }
    }
    return s;
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 10;
    if (maxW < 6 || maxW > 12) return 2;

    const auto rt = oneesan::twocell::make_rank_tables();
    const auto st = oneesan::twocell::make_stationary_rank_tables(rt);
    std::vector<std::vector<Word>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 6; W <= maxW; ++W) {
        for (int steps = 2; steps <= std::min(4, W - 3); ++steps) {
            const auto expected = expected_fusion_size_histogram(W, steps, rt);
            Rank expected_components = 0;
            Rank expected_states = 0;
            Rank max_block = 0;
            for (const auto& [size, count] : expected) {
                expected_components += count;
                expected_states += size * count;
                max_block = std::max(max_block, size);
            }
            const Rank power_components = Rank(1) << (W - steps - 3);
            if (expected_components != power_components || expected_states != st.total[W])
                fail("fusion theory partition formula W=" + std::to_string(W));

            for (int start = 0; start + steps <= W - 3; ++start) {
                const auto actual = actual_fusion_size_histogram(
                    W, start, steps, words, rt, st);
                if (actual != expected)
                    fail("fusion component histogram W=" + std::to_string(W) +
                         " start=" + std::to_string(start) +
                         " steps=" + std::to_string(steps));
            }

            std::cout << "W=" << W
                      << " steps=" << steps
                      << " outer_bits=" << (W - steps - 3)
                      << " components=" << expected_components
                      << " states=" << expected_states
                      << " max_block_states=" << max_block
                      << " position_independent=OK"
                      << " size_depends_only_outer_popcount=OK\n";
        }
    }

    const int W = 28;
    const Rank r28 = st.total[W];
    for (int steps = 2; steps <= 6; ++steps) {
        const int outer_bits = W - steps - 3;
        const Rank blocks = Rank(1) << outer_bits;
        const Rank max_states = fusion_block_size(steps, outer_bits, rt);
        const double avg_states = double(r28) / double(blocks);
        std::cout << "W=28_fusion steps=" << steps
                  << " outer_bits=" << outer_bits
                  << " blocks=" << blocks
                  << " avg_block_states=" << avg_states
                  << " avg_block_KiB_u32=" << avg_states * 4.0 / 1024.0
                  << " max_block_states=" << max_states
                  << " max_block_MiB_u32="
                  << double(max_states * 4ULL) / double(1ULL << 20)
                  << "\n";
    }

    for (Rank kib : {64ULL, 128ULL, 192ULL, 228ULL, 256ULL}) {
        const CapacityStats s = capacity_stats(W, 2, kib * 1024ULL, rt);
        const double block_fraction = double(s.fit_blocks) / double(s.all_blocks);
        const double state_fraction = double(s.fit_states) / double(s.all_states);
        const double relative_traffic = 1.0 - 0.5 * state_fraction;
        std::cout << "W=28_fusion2_capacity KiB=" << kib
                  << " max_outer_ones=" << s.max_outer_ones
                  << " fit_block_fraction=" << block_fraction
                  << " fit_state_fraction=" << state_fraction
                  << " HBM_relative_vs_two_pass=" << relative_traffic
                  << " HBM_reduction=" << (1.0 - relative_traffic)
                  << "\n";
    }

    std::cout << "ALL_OK stationary_multistep_fusion_blocks=1\n";
    return 0;
}
