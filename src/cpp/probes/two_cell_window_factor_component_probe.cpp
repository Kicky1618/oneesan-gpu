#pragma push_macro("main")
#undef main
#define main two_cell_component_label_codec_probe_main_unused
#include "two_cell_component_label_codec_probe.cpp"
#pragma pop_macro("main")

namespace {

std::uint32_t insert_support4(std::uint32_t outer, int start, std::uint32_t local) {
    const std::uint32_t lo = outer & oneesan::twocell::low_mask(start);
    const std::uint32_t hi = outer >> start;
    return lo | ((local & 15u) << start) | (hi << (start + 4));
}

Word materialize_support_primitive(
    std::uint32_t support,
    int len,
    int occupied,
    Rank primitive_rank_value,
    const oneesan::twocell::RankTables& tables
) {
    const std::uint32_t left = oneesan::twocell::primitive_left_unrank(
        support, len, occupied, primitive_rank_value, tables);
    Word w(static_cast<std::size_t>(len), N);
    for (int p = 0; p < len; ++p) {
        const std::uint32_t bit = std::uint32_t(1) << p;
        if (!(support & bit)) continue;
        w[p] = (left & bit) ? L : R;
    }
    return w;
}

Rank total_label_primitive_lut_entries(
    int W,
    const oneesan::twocell::RankTables& tables
) {
    Rank z = 0;
    const int label_len = W - 2;
    for (int occupied = 1; occupied <= label_len; occupied += 2)
        z += tables.primitive[occupied][1];
    return z;
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 14;
    if (maxW < 6 || maxW > 15) return 2;

    const auto tables = oneesan::twocell::make_rank_tables();
    std::vector<std::vector<Word>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 6; W <= maxW; ++W) {
        const int len = W - 2;
        const int outer_bits = W - 6;
        Rank expected_groups = Rank(1) << (W - 3);
        Rank worst_group = 0;
        Rank best_group = Rank(-1);

        for (int i = 1; i <= W - 5; ++i) {
            const int start = i - 1;
            std::set<Word> generated;
            Rank groups = 0;
            Rank components = 0;

            for (std::uint32_t outer = 0;
                 outer < (std::uint32_t(1) << outer_bits); ++outer) {
                const int k = oneesan::twocell::popcount32(outer);
                for (std::uint32_t local = 0; local < 16; ++local) {
                    const int occupied = k + oneesan::twocell::popcount32(local);
                    if (occupied <= 0 || !(occupied & 1)) continue;
                    const Rank pc = tables.primitive[occupied][1];
                    if (!pc) fail("window factor zero primitive count");
                    ++groups;
                    worst_group = std::max(worst_group, pc);
                    best_group = std::min(best_group, pc);

                    const std::uint32_t support = insert_support4(outer, start, local);
                    if (oneesan::twocell::popcount32(support) != occupied)
                        fail("window factor support popcount");
                    for (Rank pr = 0; pr < pc; ++pr) {
                        const Word w = materialize_support_primitive(
                            support, len, occupied, pr, tables);
                        if (!valid_word(w) || !generated.insert(w).second)
                            fail("window factor generated label W=" + std::to_string(W));
                        ++components;
                    }
                }
            }

            if (groups != expected_groups)
                fail("window factor group count W=" + std::to_string(W));
            if (components != words[W - 2].size() ||
                generated != std::set<Word>(words[W - 2].begin(), words[W - 2].end()))
                fail("window factor label coverage W=" + std::to_string(W));
        }

        std::cout << "W=" << W
                  << " labels=" << words[W - 2].size()
                  << " outer_bits=" << outer_bits
                  << " valid_support_groups=" << expected_groups
                  << " avg_components_per_group="
                  << double(words[W - 2].size()) / double(expected_groups)
                  << " min_group=" << best_group
                  << " max_group=" << worst_group
                  << " support_unrank_per_component=0"
                  << " support=outer_mask+local4"
                  << " OK\n";
    }

    const int W = 28;
    const Rank labels = oneesan::twocell::component_label_count(W, tables);
    const Rank groups = Rank(1) << (W - 3);
    const Rank primitive_entries = total_label_primitive_lut_entries(W, tables);
    std::cout << "W=28_theory labels=" << labels
              << " outer_bits=" << (W - 6)
              << " valid_support_groups=" << groups
              << " avg_components_per_group=" << double(labels) / double(groups)
              << " max_group=" << tables.primitive[25][1]
              << " primitive_lut_entries=" << primitive_entries
              << " primitive_lut_MiB="
              << double(primitive_entries * sizeof(std::uint32_t)) / double(1ULL << 20)
              << "\n";
    std::cout << "ALL_OK window_factor_component_labels=1\n";
    return 0;
}
