#pragma push_macro("main")
#undef main
#define main two_cell_stationary_rank_probe_main_unused
#include "two_cell_stationary_rank_probe.cpp"
#pragma pop_macro("main")

#include "../../common/two_cell_stationary_rank.hpp"

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 13;
    if (maxW < 5 || maxW > 15) return 2;

    const auto rt = oneesan::twocell::make_rank_tables();
    const auto st = oneesan::twocell::make_stationary_rank_tables(rt);
    std::vector<std::vector<Word>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 5; W <= maxW; ++W) {
        Rank components = 0;
        Rank coordinates = 0;
        Rank support_base_evaluations = 0;
        Rank max_distinct_supports = 0;

        for (int i = 0; i <= W - 4; ++i) {
            for (const Word& u : words[W - 2]) {
                const auto label = stationary_pack(Key{'C', u});
                const auto bases = oneesan::twocell::stationary_component_bases(
                    label.support, W, i, rt, st);
                const auto sources = packed_direct_component_sources(pack_word(u), W, i);

                std::set<std::pair<char, std::uint32_t>> distinct;
                for (int q = 0; q < sources.size; ++q) {
                    const Key src = unpack_key(sources.value[q]);
                    const auto p = stationary_pack(src);
                    const Rank primitive = oneesan::twocell::primitive_rank(
                        p.support, p.left, static_cast<int>(src.w.size()), rt);
                    const Rank expected = oneesan::twocell::stationary_rank_with_primitive(
                        p, W, i, primitive, rt, st);
                    const Rank cached = oneesan::twocell::stationary_component_source_base(
                        bases, q) + primitive;
                    if (expected != cached)
                        fail("stationary cached support base W=" + std::to_string(W) +
                             " i=" + std::to_string(i));

                    std::uint32_t support = p.support;
                    if (p.type) support = oneesan::twocell::stationary_c_support(support, i);
                    distinct.insert({src.type, support});
                    ++coordinates;
                }

                const Rank expected_bases = sources.size == 1 ? 1 :
                    (sources.size == 3 ? 3 : 5);
                if (distinct.size() != expected_bases)
                    fail("stationary distinct support count W=" + std::to_string(W));
                support_base_evaluations += expected_bases;
                max_distinct_supports = std::max<Rank>(max_distinct_supports, distinct.size());
                ++components;
            }
        }

        std::cout << "W=" << W
                  << " components=" << components
                  << " coordinates=" << coordinates
                  << " support_base_evaluations=" << support_base_evaluations
                  << " avg_support_bases_per_component="
                  << double(support_base_evaluations) / double(components)
                  << " coordinate_per_support_base="
                  << double(coordinates) / double(support_base_evaluations)
                  << " max_distinct_supports=" << max_distinct_supports
                  << " support_bases_depend_on_label_support_only=OK\n";
    }

    const Rank m26 = 47337954326ULL;
    std::cout << "W=28_theory support_bases_per_component_average=3"
              << " support_base_evaluations_per_support_tile<=5"
              << " components=" << m26
              << "\n";
    std::cout << "ALL_OK stationary_support_base_cache=1\n";
    return 0;
}
