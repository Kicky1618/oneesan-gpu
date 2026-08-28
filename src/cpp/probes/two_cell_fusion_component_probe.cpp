#pragma push_macro("main")
#undef main
#define main two_cell_direct_component_probe_main_unused
#include "two_cell_direct_component_probe.cpp"
#pragma pop_macro("main")

#include "../../common/two_cell_fusion_component.cuh"

namespace {

Word fusion_label_word(const oneesan::twocell::PackedWord& p) {
    Word w(static_cast<std::size_t>(p.len), N);
    for (int i = 0; i < p.len; ++i) {
        const std::uint32_t bit = std::uint32_t(1) << i;
        if (!(p.support & bit)) continue;
        w[static_cast<std::size_t>(i)] = (p.left & bit) ? L : R;
    }
    return w;
}

oneesan::twocell::PackedKey fusion_key(const PackedKey& k) {
    return oneesan::twocell::PackedKey{
        k.w.support, k.w.left,
        static_cast<std::uint8_t>(k.type == 'C')};
}

} // namespace

int main(int argc, char** argv) {
    constexpr int steps = 2;
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 13;
    if (maxW < 6 || maxW > 15) return 2;

    const auto rt = oneesan::twocell::make_rank_tables();
    std::vector<std::vector<Word>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 6; W <= maxW; ++W) {
        const int outer_bits = W - steps - 3;
        const Rank expected_components = words[W - 2].size();
        for (int start = 0; start + steps <= W - 3; ++start) {
            std::set<Word> labels;
            Rank component_sum = 0;
            Rank source_checks = 0;
            Rank max_block_components = 0;
            for (std::uint32_t outer = 0;
                 outer < (std::uint32_t(1) << outer_bits); ++outer) {
                const int o = oneesan::twocell::popcount32(outer);
                const Rank nc = oneesan::twocell::fusion_component_count(
                    steps, o, rt);
                const Rank block_states = oneesan::twocell::fusion_block_size(
                    steps, o, rt);
                max_block_components = std::max(max_block_components, nc);
                component_sum += nc;
                for (Rank cr = 0; cr < nc; ++cr) {
                    const auto label = oneesan::twocell::fusion_component_unrank(
                        cr, outer, W, start, steps, rt);
                    if (!label.valid || !valid_word(fusion_label_word(label.word)))
                        fail("fusion component invalid label W=" + std::to_string(W));
                    if (oneesan::twocell::fusion_component_outer_mask(
                            label.word, start, steps) != outer)
                        fail("fusion component outer mismatch");
                    if (!labels.insert(fusion_label_word(label.word)).second)
                        fail("fusion component duplicate label");

                    for (int phase = 0; phase < steps; ++phase) {
                        const int active = start + phase;
                        const auto src = packed_direct_component_sources(
                            PackedWord{label.word.support, label.word.left, label.word.len},
                            W, active);
                        if (src.size <= 0)
                            fail("fusion component source reconstruction");
                        for (int q = 0; q < src.size; ++q) {
                            const PackedKey pk = src.value[q];
                            const auto key = fusion_key(pk);
                            if (oneesan::twocell::fusion_outer_mask_at(
                                    key, start, steps, active) != outer)
                                fail("fusion component source escaped block");
                            const Rank lr = oneesan::twocell::fusion_local_rank_at(
                                key, W, start, steps, active, o, rt);
                            if (lr >= block_states)
                                fail("fusion component local rank range");
                            ++source_checks;
                        }
                    }
                }
            }
            if (component_sum != expected_components ||
                labels != std::set<Word>(words[W - 2].begin(), words[W - 2].end()))
                fail("fusion component label coverage W=" + std::to_string(W));

            std::cout << "W=" << W
                      << " start=" << start
                      << " outer_blocks=" << (Rank(1) << outer_bits)
                      << " components=" << component_sum
                      << " max_components_per_block=" << max_block_components
                      << " source_checks=" << source_checks
                      << " component_table_bytes=0 local3_support=1 OK\n";
        }
    }

    const int W = 28;
    const int outer_bits = W - 5;
    Rank component_sum = 0;
    Rank weighted_max = 0;
    for (int o = 0; o <= outer_bits; ++o) {
        const Rank nc = oneesan::twocell::fusion_component_count(steps, o, rt);
        component_sum += rt.choose[outer_bits][o] * nc;
        weighted_max = std::max(weighted_max, nc);
    }
    if (component_sum != 47337954326ULL) fail("fusion W28 component sum");
    std::cout << "W=28_theory outer_blocks=" << (Rank(1) << outer_bits)
              << " components_per_phase=" << component_sum
              << " max_components_per_block=" << weighted_max
              << " component_table_bytes=0\n";
    std::cout << "ALL_OK fusion_component_local_codec=1\n";
    return 0;
}
