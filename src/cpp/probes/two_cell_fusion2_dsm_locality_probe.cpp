#pragma push_macro("main")
#undef main
#define main two_cell_fusion_component_probe_main_unused
#include "two_cell_fusion_component_probe.cpp"
#pragma pop_macro("main")

namespace {

int primitive_slice_owner(Rank primitive, Rank count, int cluster) {
    if (!count || primitive >= count || cluster <= 0) return -1;
    int owner = static_cast<int>((primitive * Rank(cluster)) / count);
    if (owner >= cluster) owner = cluster - 1;
    return owner;
}

Rank packed_primitive_rank(
    const PackedKey& k,
    int W,
    const oneesan::twocell::RankTables& rt
) {
    const int len = k.type == 'C' ? W - 2 : W - 1;
    return oneesan::twocell::primitive_rank(k.w.support, k.w.left, len, rt);
}

Rank packed_primitive_count(
    const PackedKey& k,
    const oneesan::twocell::RankTables& rt
) {
    const int occupied = oneesan::twocell::popcount32(k.w.support);
    return oneesan::twocell::primitive_count_for_occupied(occupied, rt);
}

} // namespace

int main(int argc, char** argv) {
    constexpr int steps = 2;
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 13;
    if (maxW < 6 || maxW > 15) return 2;

    const auto rt = oneesan::twocell::make_rank_tables();

    for (int W = 6; W <= maxW; ++W) {
        const int outer_bits = W - 5;
        for (int cluster : {2, 4, 8}) {
            Rank components = 0;
            Rank sources = 0;
            Rank local_sources = 0;
            Rank base3_sources = 0;
            Rank base3_local = 0;
            Rank base4_sources = 0;
            Rank base4_local = 0;

            for (int start = 0; start + steps <= W - 3; ++start) {
                for (std::uint32_t outer = 0;
                     outer < (std::uint32_t(1) << outer_bits); ++outer) {
                    const int o = oneesan::twocell::popcount32(outer);
                    const Rank nc = oneesan::twocell::fusion_component_count(
                        steps, o, rt);

                    for (Rank cr = 0; cr < nc; ++cr) {
                        const auto label = oneesan::twocell::fusion_component_unrank(
                            cr, outer, W, start, steps, rt);
                        if (!label.valid) fail("DSM locality label");
                        const int label_occupied = oneesan::twocell::popcount32(
                            label.word.support);
                        const Rank label_count =
                            oneesan::twocell::primitive_count_for_occupied(
                                label_occupied, rt);
                        const int owner = primitive_slice_owner(
                            label.primitive, label_count, cluster);
                        if (owner < 0) fail("DSM locality owner");

                        for (int phase = 0; phase < steps; ++phase) {
                            const int active = start + phase;
                            const auto src = packed_direct_component_sources(
                                PackedWord{label.word.support, label.word.left, label.word.len},
                                W, active);
                            if (src.size <= 0) fail("DSM locality sources");

                            const bool retained = oneesan::twocell::symbol(
                                oneesan::twocell::PackedWord{
                                    label.word.support, label.word.left, label.word.len},
                                active) != oneesan::twocell::TC_N;
                            const bool xN_deep = retained &&
                                oneesan::twocell::symbol(
                                    oneesan::twocell::PackedWord{
                                        label.word.support, label.word.left, label.word.len},
                                    active + 1) == oneesan::twocell::TC_N;

                            for (int q = 0; q < src.size; ++q) {
                                const Rank pr = packed_primitive_rank(src.value[q], W, rt);
                                const Rank pc = packed_primitive_count(src.value[q], rt);
                                const int source_owner = primitive_slice_owner(pr, pc, cluster);
                                const bool local = source_owner == owner;
                                ++sources;
                                local_sources += local;

                                if (retained && q < 3) {
                                    ++base3_sources;
                                    base3_local += local;
                                    if (pr != label.primitive || pc != label_count)
                                        fail("DSM locality retained base3 signature");
                                }
                                if (xN_deep && q == 3) {
                                    ++base4_sources;
                                    base4_local += local;
                                    if (pr != label.primitive || pc != label_count)
                                        fail("DSM locality xN base4 signature");
                                }
                            }
                            ++components;
                        }
                    }
                }
            }

            if (base3_sources != base3_local || base4_sources != base4_local)
                fail("DSM locality guaranteed-local coordinates");

            std::cout << "W=" << W
                      << " cluster=" << cluster
                      << " component_phases=" << components
                      << " source_accesses=" << sources
                      << " local_source_accesses=" << local_sources
                      << " local_fraction=" << double(local_sources) / double(sources)
                      << " retained_base3_local_fraction="
                      << (base3_sources ? double(base3_local) / double(base3_sources) : 1.0)
                      << " xN_fourth_local_fraction="
                      << (base4_sources ? double(base4_local) / double(base4_sources) : 1.0)
                      << " contiguous_global_sector_slices=1"
                      << " OK\n";
        }
    }

    std::cout << "ALL_OK primitive_sliced_DSM_locality=1\n";
    return 0;
}
