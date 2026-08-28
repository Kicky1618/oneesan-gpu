#pragma push_macro("main")
#undef main
#define main two_cell_fusion_unrank_probe_main_unused
#include "two_cell_fusion_unrank_probe.cpp"
#pragma pop_macro("main")

#include "../../common/two_cell_fusion_sectors.hpp"

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 11;
    if (maxW < 6 || maxW > 13) return 2;

    const auto rt = oneesan::twocell::make_rank_tables();
    const auto st = oneesan::twocell::make_stationary_rank_tables(rt);

    for (int W = 6; W <= maxW; ++W) {
        for (int steps = 1; steps <= std::min(4, W - 3); ++steps) {
            for (int start = 0; start + steps <= W - 3; ++start) {
                const int outer_bits = W - steps - 3;
                Rank blocks = 0;
                Rank states = 0;
                Rank sectors = 0;
                for (std::uint32_t outer = 0;
                     outer < (std::uint32_t(1) << outer_bits); ++outer) {
                    const int o = oneesan::twocell::popcount32(outer);
                    const Rank n = oneesan::twocell::fusion_block_size(steps, o, rt);
                    Rank covered = 0;
                    Rank nonempty = 0;

                    for (int q = 0; q < oneesan::twocell::fusion_sector_count(steps); ++q) {
                        const auto sec = oneesan::twocell::fusion_sector(
                            q, outer, W, start, steps, rt, st);
                        if (!sec.valid) continue;
                        ++nonempty;
                        if (sec.local_base != covered)
                            fail("fusion sector local contiguity W=" + std::to_string(W));

                        for (Rank p = 0; p < sec.count; ++p) {
                            const Rank lr = sec.local_base + p;
                            for (int active = start; active <= start + steps; ++active) {
                                const auto d = oneesan::twocell::fusion_local_unrank_at(
                                    lr, outer, W, start, steps, active, rt);
                                if (!d.valid || d.primitive != p)
                                    fail("fusion sector unrank primitive");
                                const Rank gr = oneesan::twocell::stationary_rank_with_primitive(
                                    d.key, W, active, d.primitive, rt, st);
                                if (gr != sec.global_base + p)
                                    fail("fusion sector global interval W=" +
                                         std::to_string(W) + " active=" +
                                         std::to_string(active));
                            }
                            ++states;
                        }
                        covered += sec.count;
                        ++sectors;
                    }

                    if (covered != n)
                        fail("fusion sector block coverage W=" + std::to_string(W));
                    if (nonempty != oneesan::twocell::fusion_sector_nonempty_count(
                            outer, W, start, steps, rt, st))
                        fail("fusion sector nonempty count");
                    ++blocks;
                }

                std::cout << "W=" << W
                          << " start=" << start
                          << " steps=" << steps
                          << " blocks=" << blocks
                          << " state_active_checks=" << states
                          << " nonempty_sectors=" << sectors
                          << " max_sector_slots="
                          << oneesan::twocell::fusion_sector_count(steps)
                          << " load_store_same_global_intervals=OK"
                          << " per_state_unrank_for_copy=0\n";
            }
        }
    }

    std::cout << "W=28_fusion2_theory max_sector_slots=20"
              << " block_copy_address_builds_per_outer<=20"
              << " start_end_stationary_global_bases_identical=1\n";
    std::cout << "ALL_OK fusion_contiguous_sector_copy=1\n";
    return 0;
}
