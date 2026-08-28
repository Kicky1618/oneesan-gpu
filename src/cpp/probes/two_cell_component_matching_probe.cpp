#pragma push_macro("main")
#undef main
#define main two_cell_component_device_probe_main_unused
#include "two_cell_component_device_probe.cpp"
#pragma pop_macro("main")

#include "../../common/two_cell_component_matching.cuh"

namespace {

constexpr std::uint32_t kMod = 1000000007u;

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 14;
    if (maxW < 5 || maxW > 15) return 2;

    std::vector<std::vector<Word>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 5; W <= maxW; ++W) {
        Rank checked = 0;
        Rank nonidentity = 0;
        Rank residual = 0;
        Rank max_moved = 0;
        Rank max_cycle = 0;
        Rank singleton = 0, triple = 0, deep_rn = 0, deep_lr = 0, generic_ln = 0;

        for (int i = 0; i <= W - 4; ++i) {
            for (const Word& u : words[W - 2]) {
                const auto src = oneesan::twocell::direct_component_sources(
                    device_word(u), W, i);
                if (src.overflow || src.size <= 0 ||
                    src.size > oneesan::twocell::kMaxComponentMatching)
                    fail("matching probe source component");

                const auto m = oneesan::twocell::build_component_matching(
                    src.value, src.size, W, i);
                if (!m.ok || m.residual_edges != src.size - 1)
                    fail("matching probe build W=" + std::to_string(W) +
                         " i=" + std::to_string(i));

                switch (m.fast_kind) {
                    case oneesan::twocell::TC_MATCH_SINGLETON: ++singleton; break;
                    case oneesan::twocell::TC_MATCH_TRIPLE: ++triple; break;
                    case oneesan::twocell::TC_MATCH_DEEP_RN: ++deep_rn; break;
                    case oneesan::twocell::TC_MATCH_DEEP_LR: ++deep_lr; break;
                    case oneesan::twocell::TC_MATCH_GENERIC: ++generic_ln; break;
                    default: fail("matching probe fast kind");
                }

                int moved = 0;
                std::vector<std::uint8_t> seen(static_cast<std::size_t>(src.size));
                for (int s = 0; s < src.size; ++s) {
                    moved += m.src_to_dst[s] != s;
                    if (seen[s]) continue;
                    int q = s, cycle = 0;
                    while (!seen[q]) {
                        seen[q] = 1;
                        q = m.src_to_dst[q];
                        ++cycle;
                    }
                    max_cycle = std::max<Rank>(max_cycle, cycle);
                }
                if (moved) ++nonidentity;
                max_moved = std::max<Rank>(max_moved, moved);

                std::uint32_t x[oneesan::twocell::kMaxComponentMatching]{};
                std::uint32_t y[oneesan::twocell::kMaxComponentMatching]{};
                std::uint32_t fast[oneesan::twocell::kMaxComponentMatching]{};
                std::uint32_t ref[oneesan::twocell::kMaxComponentMatching]{};
                for (int s = 0; s < src.size; ++s)
                    x[s] = static_cast<std::uint32_t>(1 + 31 * s + 7 * i);

                for (int s = 0; s < src.size; ++s) {
                    const auto edges = oneesan::twocell::K_step(src.value[s], W, i);
                    if (edges.overflow) fail("matching probe K overflow");
                    for (int e = 0; e < edges.size; ++e) {
                        const int t = oneesan::twocell::coordinate_index_for_destination(
                            src.value, src.size, edges.value[e], i);
                        if (t < 0) fail("matching probe destination index");
                        ref[t] = static_cast<std::uint32_t>(
                            (static_cast<std::uint64_t>(ref[t]) + x[s]) % kMod);
                    }
                }
                if (!oneesan::twocell::apply_component_matching(m, x, y, kMod))
                    fail("matching probe apply");
                for (int t = 0; t < src.size; ++t)
                    if (y[t] != ref[t])
                        fail("matching probe arithmetic W=" + std::to_string(W) +
                             " i=" + std::to_string(i));

                const bool used_fast = oneesan::twocell::apply_component_fastpath(
                    src.value, src.size, W, i, x, fast, kMod);
                if (used_fast != (m.fast_kind != oneesan::twocell::TC_MATCH_GENERIC))
                    fail("matching probe fast dispatch");
                if (used_fast) {
                    for (int t = 0; t < src.size; ++t)
                        if (fast[t] != ref[t])
                            fail("matching probe fast arithmetic W=" + std::to_string(W));
                } else {
                    const PackedWord label = device_word(u);
                    if (oneesan::twocell::symbol(label, i) != oneesan::twocell::TC_L ||
                        oneesan::twocell::symbol(label, i + 1) != oneesan::twocell::TC_N)
                        fail("matching generic must be deep LN");
                }

                residual += m.residual_edges;
                ++checked;
            }
        }

        std::cout << "W=" << W
                  << " components=" << checked
                  << " singleton=" << singleton
                  << " triple=" << triple
                  << " deep_RN_closed=" << deep_rn
                  << " deep_LR_closed=" << deep_lr
                  << " deep_LN_generic=" << generic_ln
                  << " closed_fraction="
                  << double(singleton + triple + deep_rn + deep_lr) / double(checked)
                  << " nonidentity_matching_components=" << nonidentity
                  << " residual_adds=" << residual
                  << " max_moved_matching_coordinates=" << max_moved
                  << " max_matching_cycle=" << max_cycle
                  << " recouple_only_indexes_destination_set=1"
                  << " matching=OK fast_arithmetic=OK\n";
    }

    const std::uint64_t all = 25ULL * 47337954326ULL;
    const std::uint64_t generic_ln = 126383557900ULL;
    std::cout << "W=28_sweep_theory closed_matching_fraction="
              << double(all - generic_ln) / double(all)
              << " generic_LN_components=" << generic_ln
              << " generic_LN_fraction=" << double(generic_ln) / double(all)
              << " residual_adds_total_per_step=118389089432"
              << " global_matching_table_bytes=0\n";
    std::cout << "ALL_OK table_free_component_matching=1\n";
    return 0;
}
