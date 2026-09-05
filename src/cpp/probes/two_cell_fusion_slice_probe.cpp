#pragma push_macro("main")
#undef main
#define main two_cell_fusion_sector_probe_main_unused
#include "two_cell_fusion_sector_probe.cpp"
#pragma pop_macro("main")

#include "../../common/two_cell_fusion_slices.hpp"

int main(int argc, char** argv) {
    constexpr int steps = 2;
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 11;
    if (maxW < 6 || maxW > 13) return 2;

    const auto rt = oneesan::twocell::make_rank_tables();
    const auto st = oneesan::twocell::make_stationary_rank_tables(rt);

    for (int W = 6; W <= maxW; ++W) {
        const int outer_bits = W - 5;
        for (int start = 0; start + steps <= W - 3; ++start) {
            for (int owners : {2, 4, 8}) {
                Rank checks = 0;
                Rank max_owner_states = 0;
                for (std::uint32_t outer = 0;
                     outer < (std::uint32_t(1) << outer_bits); ++outer) {
                    const int o = oneesan::twocell::popcount32(outer);
                    const Rank n = oneesan::twocell::fusion_block_size(steps, o, rt);

                    oneesan::twocell::FusionSector sectors[20]{};
                    for (int q = 0; q < 20; ++q)
                        sectors[q] = oneesan::twocell::fusion_sector(
                            q, outer, W, start, steps, rt, st);

                    Rank owner_total[8]{};
                    for (int owner = 0; owner < owners; ++owner) {
                        Rank total = 0;
                        for (int q = 0; q < 20; ++q) {
                            if (!sectors[q].valid) continue;
                            const Rank base = oneesan::twocell::fusion_slice_local_base(
                                sectors, q, owner, owners);
                            if (base != total) fail("fusion slice prefix");
                            total += oneesan::twocell::primitive_slice_count(
                                sectors[q].count, owner, owners);
                        }
                        owner_total[owner] = total;
                        max_owner_states = std::max(max_owner_states, total);
                    }
                    Rank total_partition = 0;
                    for (int owner = 0; owner < owners; ++owner)
                        total_partition += owner_total[owner];
                    if (total_partition != n) fail("fusion slice total partition");

                    for (Rank lr = 0; lr < n; ++lr) {
                        for (int active = start; active <= start + steps; ++active) {
                            const auto d = oneesan::twocell::fusion_local_unrank_at(
                                lr, outer, W, start, steps, active, rt);
                            if (!d.valid) fail("fusion slice unrank");
                            const int q = oneesan::twocell::fusion_sector_index_at(
                                d.key, start, steps, active);
                            if (q < 0 || q >= 20 || !sectors[q].valid)
                                fail("fusion slice sector index");
                            if (sectors[q].local_base + d.primitive != lr)
                                fail("fusion slice local rank identity");

                            const int owner = oneesan::twocell::primitive_slice_owner(
                                d.primitive, sectors[q].count, owners);
                            if (owner < 0 || owner >= owners)
                                fail("fusion slice owner");
                            const Rank begin = oneesan::twocell::primitive_slice_begin(
                                sectors[q].count, owner, owners);
                            const Rank end = oneesan::twocell::primitive_slice_end(
                                sectors[q].count, owner, owners);
                            if (d.primitive < begin || d.primitive >= end)
                                fail("fusion slice range");
                            const Rank local = oneesan::twocell::fusion_slice_local_base(
                                sectors, q, owner, owners) + d.primitive - begin;
                            if (local >= owner_total[owner])
                                fail("fusion slice CTA offset");
                            ++checks;
                        }
                    }
                }
                std::cout << "W=" << W
                          << " start=" << start
                          << " owners=" << owners
                          << " checks=" << checks
                          << " max_owner_states=" << max_owner_states
                          << " sector_plus_primitive_address=OK"
                          << " fusion_local_rank_needed_in_DSM=0\n";
            }
        }
    }

    std::cout << "ALL_OK fusion_primitive_slice_addressing=1\n";
    return 0;
}
