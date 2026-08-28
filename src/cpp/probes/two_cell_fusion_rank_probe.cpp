#pragma push_macro("main")
#undef main
#define main two_cell_multistep_fusion_probe_main_unused
#include "two_cell_multistep_fusion_probe.cpp"
#pragma pop_macro("main")

#include "../../common/two_cell_fusion_rank.hpp"

namespace {

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
            for (int start = 0; start + steps <= W - 3; ++start) {
                const int outer_bits = W - steps - 3;
                std::map<std::uint32_t, std::vector<std::uint8_t>> seen;

                for (const Key& k : q_basis(W, start, words)) {
                    const auto p = stationary_pack(k);
                    const std::uint32_t outer = oneesan::twocell::fusion_outer_mask_at(
                        p, start, steps, start);
                    if (outer >= (std::uint32_t(1) << outer_bits))
                        fail("fusion rank outer range");
                    const int o = oneesan::twocell::popcount32(outer);
                    const Rank block = oneesan::twocell::fusion_block_size(steps, o, rt);
                    auto& mark = seen[outer];
                    if (mark.empty()) mark.assign(static_cast<std::size_t>(block), 0);
                    const Rank r = oneesan::twocell::fusion_local_rank_at(
                        p, W, start, steps, start, o, rt);
                    if (r >= block || mark[static_cast<std::size_t>(r)]++)
                        fail("fusion rank source collision W=" + std::to_string(W));
                }

                if (seen.size() != (std::size_t(1) << outer_bits))
                    fail("fusion rank missing outer mask");
                for (const auto& [outer, mark] : seen)
                    for (std::uint8_t x : mark)
                        if (x != 1) fail("fusion rank local gap");

                Rank edge_checks = 0;
                for (int active = start; active < start + steps; ++active) {
                    for (const Key& s : q_basis(W, active, words)) {
                        const auto ps = stationary_pack(s);
                        const std::uint32_t so = oneesan::twocell::fusion_outer_mask_at(
                            ps, start, steps, active);
                        const int ones = oneesan::twocell::popcount32(so);
                        const Rank sr = oneesan::twocell::fusion_local_rank_at(
                            ps, W, start, steps, active, ones, rt);
                        const Rank block = oneesan::twocell::fusion_block_size(steps, ones, rt);
                        if (sr >= block) fail("fusion rank intermediate source range");

                        for (const auto& [d, c] : K_basis(s, W, active)) {
                            if (c != 1) fail("fusion rank nonunit edge");
                            const auto pd = stationary_pack(d);
                            const std::uint32_t dout = oneesan::twocell::fusion_outer_mask_at(
                                pd, start, steps, active + 1);
                            if (dout != so)
                                fail("fusion rank edge leaves outer block W=" + std::to_string(W));
                            const Rank dr = oneesan::twocell::fusion_local_rank_at(
                                pd, W, start, steps, active + 1, ones, rt);
                            if (dr >= block)
                                fail("fusion rank intermediate destination range");
                            // Global stationary coordinates and block-local
                            // coordinates must identify the same equality relation.
                            const Rank gs = oneesan::twocell::stationary_rank(
                                ps, W, active, rt, st);
                            const Rank gd = oneesan::twocell::stationary_rank(
                                pd, W, active + 1, rt, st);
                            if ((gs == gd) != (sr == dr))
                                fail("fusion rank equality mismatch");
                            ++edge_checks;
                        }
                    }
                }

                std::cout << "W=" << W
                          << " start=" << start
                          << " steps=" << steps
                          << " outer_blocks=" << seen.size()
                          << " edge_checks=" << edge_checks
                          << " source_local_bijection=OK"
                          << " intermediate_rebase=OK"
                          << " edge_block_closure=OK\n";
            }
        }
    }

    const int W = 28;
    for (int steps = 2; steps <= 4; ++steps) {
        const int outer_bits = W - steps - 3;
        const int typical = outer_bits / 2;
        std::cout << "W=28_fusion_rank steps=" << steps
                  << " A_local_codes=" << (std::uint32_t(1) << (steps + 2))
                  << " C_local_codes=" << (std::uint32_t(1) << steps)
                  << " outer_bits=" << outer_bits
                  << " typical_outer_ones=" << typical
                  << " typical_block_states="
                  << oneesan::twocell::fusion_block_size(steps, typical, rt)
                  << " global_dictionary_bytes=0\n";
    }
    std::cout << "ALL_OK fusion_local_rank=1\n";
    return 0;
}
