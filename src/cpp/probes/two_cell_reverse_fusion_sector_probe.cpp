#pragma push_macro("main")
#undef main
#define main two_cell_reverse_stationary_probe_main_unused
#include "two_cell_reverse_stationary_probe.cpp"
#pragma pop_macro("main")

#include "../../common/two_cell_fusion_sectors.hpp"

#include <map>
#include <set>

namespace {

using oneesan::twocell::FusionSector;

oneesan::twocell::PackedKey pack_reverse_key(const Key& k) {
    return reverse_stationary_pack(k);
}

void check_sector_address(
    const Key& key,
    int W,
    int start,
    int steps,
    int active,
    std::uint32_t outer,
    const oneesan::twocell::RankTables& rt,
    const oneesan::twocell::StationaryRankTables& st
) {
    const auto p = pack_reverse_key(key);
    if (oneesan::twocell::fusion_outer_mask_at(
            p, start, steps, active) != outer)
        fail("reverse fusion outer mismatch");

    const int sector_index = oneesan::twocell::fusion_sector_index_at(
        p, start, steps, active);
    const int sector_count = (1 << (steps + 2)) + (1 << steps);
    if (sector_index < 0 || sector_index >= sector_count)
        fail("reverse fusion sector index range");

    const FusionSector sector = oneesan::twocell::fusion_sector(
        sector_index, outer, W, start, steps, rt, st);
    if (!sector.valid) fail("reverse fusion invalid sector");

    const int len = p.type ? W - 2 : W - 1;
    const auto primitive = oneesan::twocell::primitive_rank(
        p.support, p.left, len, rt);
    if (primitive >= sector.count)
        fail("reverse fusion primitive outside sector");

    const auto global = oneesan::twocell::stationary_rank_with_primitive(
        p, W, active, primitive, rt, st);
    if (global != sector.global_base + primitive)
        fail("reverse fusion stationary sector address mismatch");
}

} // namespace

int main(int argc, char** argv) {
    constexpr int steps = 2;
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 12;
    if (maxW < 6 || maxW > 14) return 2;

    const auto rt = oneesan::twocell::make_rank_tables();
    const auto st = oneesan::twocell::make_stationary_rank_tables(rt);
    std::vector<std::vector<Word>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 6; W <= maxW; ++W) {
        Rank source_checks = 0;
        Rank edge_checks = 0;
        Rank block_checks = 0;

        for (int start = 0; start <= W - 5; ++start) {
            const int source_active = start + 2;
            const int middle_active = start + 1;
            const int final_active = start;
            const int first_pair = source_active + 1;
            const int second_pair = middle_active + 1;
            const int final_pair = final_active + 1;
            const int outer_bits = W - steps - 3;

            std::map<std::uint32_t, Rank> source_blocks;
            std::map<std::uint32_t, Rank> middle_blocks;
            std::map<std::uint32_t, Rank> final_blocks;

            const auto src_basis = reverse_q_basis(W, first_pair, words);
            for (const Key& src : src_basis) {
                const auto ps = pack_reverse_key(src);
                const std::uint32_t outer = oneesan::twocell::fusion_outer_mask_at(
                    ps, start, steps, source_active);
                ++source_blocks[outer];
                check_sector_address(
                    src, W, start, steps, source_active, outer, rt, st);
                ++source_checks;

                for (const auto& [mid, c1] : K_reverse_basis(src, W, first_pair)) {
                    if (c1 != 1) fail("reverse fusion first nonunit edge");
                    const auto pm = pack_reverse_key(mid);
                    if (oneesan::twocell::fusion_outer_mask_at(
                            pm, start, steps, middle_active) != outer)
                        fail("reverse fusion first step escaped block");
                    check_sector_address(
                        mid, W, start, steps, middle_active, outer, rt, st);
                    ++edge_checks;

                    for (const auto& [dst, c2] : K_reverse_basis(
                             mid, W, second_pair)) {
                        if (c2 != 1) fail("reverse fusion second nonunit edge");
                        const auto pd = pack_reverse_key(dst);
                        if (oneesan::twocell::fusion_outer_mask_at(
                                pd, start, steps, final_active) != outer)
                            fail("reverse fusion second step escaped block");
                        check_sector_address(
                            dst, W, start, steps, final_active, outer, rt, st);
                        ++edge_checks;
                    }
                }
            }

            for (const Key& mid : reverse_q_basis(W, second_pair, words)) {
                const auto p = pack_reverse_key(mid);
                ++middle_blocks[oneesan::twocell::fusion_outer_mask_at(
                    p, start, steps, middle_active)];
            }
            for (const Key& dst : reverse_q_basis(W, final_pair, words)) {
                const auto p = pack_reverse_key(dst);
                ++final_blocks[oneesan::twocell::fusion_outer_mask_at(
                    p, start, steps, final_active)];
            }

            for (std::uint32_t outer = 0;
                 outer < (std::uint32_t(1) << outer_bits); ++outer) {
                const int o = oneesan::twocell::popcount32(outer);
                const Rank expected = oneesan::twocell::fusion_block_size(
                    steps, o, rt);
                if (source_blocks[outer] != expected ||
                    middle_blocks[outer] != expected ||
                    final_blocks[outer] != expected)
                    fail("reverse fusion block size mismatch W=" +
                         std::to_string(W) + " start=" + std::to_string(start));
                ++block_checks;
            }
        }

        std::cout << "W=" << W
                  << " source_checks=" << source_checks
                  << " edge_checks=" << edge_checks
                  << " block_checks=" << block_checks
                  << " reverse_fusion_outer_invariant=OK"
                  << " stationary_sector_address=OK"
                  << " capacity_same_as_forward=1\n";
    }

    std::cout << "W=28_theory reverse_fusion2_sectors=20"
              << " reverse_capacity_distribution_equals_forward=1"
              << " reverse_global_permutation_bytes=0\n";
    std::cout << "ALL_OK reverse_fusion_sector_invariance=1\n";
    return 0;
}
