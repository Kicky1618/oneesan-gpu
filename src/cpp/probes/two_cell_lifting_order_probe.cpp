#pragma push_macro("main")
#undef main
#define main two_cell_stationary_lifting_probe_main_unused
#include "two_cell_stationary_lifting_probe.cpp"
#pragma pop_macro("main")

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 13;
    if (maxW < 5 || maxW > 15) return 2;

    std::vector<std::vector<Word>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 5; W <= maxW; ++W) {
        Rank components = 0;
        Rank ascending = 0;
        Rank descending = 0;
        Rank mixed = 0;
        Rank natural_forward_valid = 0;
        Rank natural_reverse_valid = 0;
        Rank max_forward_violations = 0;
        Rank max_reverse_violations = 0;

        for (int i = 0; i <= W - 4; ++i) {
            for (const Word& u : words[W - 2]) {
                const auto packed_sources = packed_direct_component_sources(pack_word(u), W, i);
                std::vector<Key> src;
                for (int q = 0; q < packed_sources.size; ++q)
                    src.push_back(unpack_key(packed_sources.value[q]));

                std::vector<DirectedEdge> shear;
                bool all_up = true;
                bool all_down = true;
                for (int s = 0; s < static_cast<int>(src.size()); ++s) {
                    for (const auto& [d, c] : K_basis(src[static_cast<std::size_t>(s)], W, i)) {
                        if (c != 1) fail("order coefficient");
                        const int t = source_index_for_destination(src, d, i);
                        if (t < 0) fail("order destination");
                        if (t == s) continue;
                        shear.push_back({s, t});
                        all_up &= s < t;
                        all_down &= s > t;
                    }
                }

                if (all_up) ++ascending;
                else if (all_down) ++descending;
                else ++mixed;

                // Direct ascending vertex processing is valid iff every shear
                // source is processed before any incoming update to that source:
                // for edge s->t we need t processed before s in the vertex order.
                Rank forward_violations = 0;
                Rank reverse_violations = 0;
                for (const auto& e : shear) {
                    if (!(e.dst < e.src)) ++forward_violations;
                    if (!(e.dst > e.src)) ++reverse_violations;
                }
                if (!forward_violations) ++natural_forward_valid;
                if (!reverse_violations) ++natural_reverse_valid;
                max_forward_violations = std::max(max_forward_violations, forward_violations);
                max_reverse_violations = std::max(max_reverse_violations, reverse_violations);
                ++components;
            }
        }

        std::cout << "W=" << W
                  << " components=" << components
                  << " all_edges_src_lt_dst=" << ascending
                  << " all_edges_src_gt_dst=" << descending
                  << " mixed=" << mixed
                  << " direct_ascending_lifting_valid=" << natural_forward_valid
                  << " direct_descending_lifting_valid=" << natural_reverse_valid
                  << " max_ascending_order_violations=" << max_forward_violations
                  << " max_descending_order_violations=" << max_reverse_violations
                  << "\n";
    }

    std::cout << "ALL_OK lifting_order_profile=1\n";
    return 0;
}
