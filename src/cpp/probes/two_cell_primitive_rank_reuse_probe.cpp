#pragma push_macro("main")
#undef main
#define main two_cell_coordinate_recoupling_probe_main_unused
#include "two_cell_coordinate_recoupling_probe.cpp"
#pragma pop_macro("main")

#include "../../common/two_cell_recoupling_rank.hpp"

namespace {

oneesan::twocell::PackedKey common_key(const Key& k) {
    oneesan::twocell::PackedKey out{};
    out.type = static_cast<std::uint8_t>(k.type == 'C');
    for (int p = 0; p < static_cast<int>(k.w.size()); ++p) {
        const std::uint32_t bit = std::uint32_t(1) << p;
        if (k.w[p] != N) out.support |= bit;
        if (k.w[p] == L) out.left |= bit;
    }
    return out;
}

Rank common_primitive(const Key& k, const oneesan::twocell::RankTables& t) {
    const auto p = common_key(k);
    return oneesan::twocell::primitive_rank(
        p.support, p.left, static_cast<int>(k.w.size()), t);
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 13;
    if (maxW < 6 || maxW > 15) return 2;

    const auto tables = oneesan::twocell::make_rank_tables();
    std::vector<std::vector<Word>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 6; W <= maxW; ++W) {
        Rank state_coordinates = 0;
        Rank baseline_primitive_calls = 0;
        Rank reused_destination_calls = 0;
        Rank label_rank_reuses = 0;
        Rank source_scan_upper = 0;

        for (int i = 1; i <= W - 5; ++i) {
            for (const Word& u : words[W - 2]) {
                const auto label = common_key(Key{'C', u});
                const Rank label_primitive = oneesan::twocell::primitive_rank(
                    label.support, label.left, W - 2, tables);
                const auto blocks = oneesan::twocell::component_blocks(
                    label.support, W, i, tables);
                const auto sources = packed_direct_component_sources(pack_word(u), W, i);

                std::set<Key> actual_destinations;
                for (int q = 0; q < sources.size; ++q) {
                    const Key s = unpack_key(sources.value[q]);
                    const auto ps = common_key(s);
                    const Rank primitive = common_primitive(s, tables);
                    const Rank sr0 = oneesan::twocell::rank_state(ps, W, i - 1, tables);
                    const Rank sr1 = oneesan::twocell::rank_with_block_primitive(
                        ps, i - 1, blocks.input_ones, blocks.input_base,
                        primitive, tables);
                    if (sr0 != sr1) fail("source primitive reuse rank");

                    // The tiled enumerator already knows label_primitive as its
                    // primitive-rank loop index. Retained components therefore
                    // need no rank scan for their first three coordinates.
                    if (u[i] != N && q < 3) {
                        if (primitive != label_primitive)
                            fail("base-three label primitive mismatch");
                        ++label_rank_reuses;
                    } else {
                        ++source_scan_upper;
                    }

                    const auto pd = oneesan::twocell::recouple_coordinate(ps, i);
                    const Key d = recouple_coordinate(s, i);
                    if (pd.support != common_key(d).support ||
                        pd.left != common_key(d).left || pd.type != common_key(d).type)
                        fail("common recouple coordinate mismatch");
                    const Rank dprimitive = common_primitive(d, tables);
                    if (dprimitive != primitive)
                        fail("destination primitive not preserved");
                    const Rank dr0 = oneesan::twocell::rank_state(pd, W, i, tables);
                    const Rank dr1 = oneesan::twocell::rank_with_block_primitive(
                        pd, i, blocks.output_ones, blocks.output_base,
                        primitive, tables);
                    if (dr0 != dr1) fail("destination primitive reuse rank");

                    actual_destinations.insert(d);
                    state_coordinates += 1;
                    baseline_primitive_calls += 2;
                }

                std::set<Key> oracle_destinations;
                for (int q = 0; q < sources.size; ++q) {
                    const Key s = unpack_key(sources.value[q]);
                    for (const auto& [d, c] : K_basis(s, W, i)) {
                        if (c != 1) fail("primitive reuse nonunit K");
                        oracle_destinations.insert(d);
                    }
                }
                if (actual_destinations != oracle_destinations)
                    fail("primitive reuse destination set mismatch");
            }
        }

        reused_destination_calls = 0;
        const double reduction = source_scan_upper
            ? double(baseline_primitive_calls) / double(source_scan_upper)
            : 0.0;
        std::cout << "W=" << W
                  << " state_coordinates=" << state_coordinates
                  << " baseline_primitive_calls=" << baseline_primitive_calls
                  << " source_scan_upper=" << source_scan_upper
                  << " destination_primitive_calls=" << reused_destination_calls
                  << " label_rank_reuses=" << label_rank_reuses
                  << " primitive_call_reduction_lower_bound=" << reduction
                  << " recoupled_rank_exact=OK\n";
    }

    const Rank dim28 = 165727043758ULL;
    const Rank m26 = 47337954326ULL;
    const Rank m25 = 16626415975ULL;
    const Rank baseline28 = 2 * dim28;
    const Rank scan28 = dim28 - 3 * (m26 - m25);
    std::cout << "W=28_theory baseline_primitive_calls_per_step=" << baseline28
              << " source_scan_upper_per_step=" << scan28
              << " destination_primitive_calls_per_step=0"
              << " call_reduction_lower_bound=" << double(baseline28) / double(scan28)
              << "\n";
    std::cout << "ALL_OK primitive_rank_recoupling_reuse=1\n";
    return 0;
}
