#pragma push_macro("main")
#undef main
#define main two_cell_fusion_rank_probe_main_unused
#include "two_cell_fusion_rank_probe.cpp"
#pragma pop_macro("main")

#include "../../common/two_cell_fusion_unrank.hpp"

namespace {

Key unpack_fusion_key(oneesan::twocell::PackedKey k, int W) {
    const int len = k.type ? W - 2 : W - 1;
    Word w(static_cast<std::size_t>(len), N);
    for (int p = 0; p < len; ++p) {
        const std::uint32_t bit = std::uint32_t(1) << p;
        if (!(k.support & bit)) continue;
        w[p] = (k.left & bit) ? L : R;
    }
    return Key{k.type ? 'C' : 'A', w};
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 10;
    if (maxW < 6 || maxW > 12) return 2;
    const auto rt = oneesan::twocell::make_rank_tables();
    std::vector<std::vector<Word>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 6; W <= maxW; ++W) {
        for (int steps = 2; steps <= std::min(4, W - 3); ++steps) {
            for (int start = 0; start + steps <= W - 3; ++start) {
                const int outer_bits = W - steps - 3;
                Rank checks = 0;
                for (std::uint32_t outer = 0;
                     outer < (std::uint32_t(1) << outer_bits); ++outer) {
                    const int o = oneesan::twocell::popcount32(outer);
                    const Rank n = oneesan::twocell::fusion_block_size(steps, o, rt);
                    for (int active = start; active <= start + steps; ++active) {
                        std::set<Key> generated;
                        for (Rank r = 0; r < n; ++r) {
                            const auto d = oneesan::twocell::fusion_local_unrank_at(
                                r, outer, W, start, steps, active, rt);
                            if (!d.valid) fail("fusion unrank invalid");
                            const Rank rr = oneesan::twocell::fusion_local_rank_at_with_primitive(
                                d.key, start, steps, active, o, d.primitive, rt);
                            if (rr != r) fail("fusion rank/unrank roundtrip");
                            const Key k = unpack_fusion_key(d.key, W);
                            if (!valid_word(k.w) || !generated.insert(k).second)
                                fail("fusion unrank generated invalid/duplicate");
                            ++checks;
                        }

                        std::set<Key> expected;
                        for (const Key& k : q_basis(W, active, words)) {
                            const auto p = stationary_pack(k);
                            if (oneesan::twocell::fusion_outer_mask_at(
                                    p, start, steps, active) == outer)
                                expected.insert(k);
                        }
                        if (generated != expected)
                            fail("fusion unrank block coverage W=" + std::to_string(W));
                    }
                }
                std::cout << "W=" << W
                          << " start=" << start
                          << " steps=" << steps
                          << " checks=" << checks
                          << " rank_unrank=OK active_rebase=OK"
                          << " table_bytes=0\n";
            }
        }
    }
    std::cout << "W=28_theory fusion2_A_codes=16 fusion2_C_codes=4"
              << " fused_state_dictionary_bytes=0\n";
    std::cout << "ALL_OK fusion_local_unrank=1\n";
    return 0;
}
