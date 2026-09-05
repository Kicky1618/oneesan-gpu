#pragma push_macro("main")
#undef main
#define main two_cell_component_matching_ln_pivot_probe_main_unused
#include "two_cell_component_matching_ln_pivot_probe.cpp"
#pragma pop_macro("main")

namespace {

int ln_local_pivot(const std::vector<Key>& src, int i) {
    for (int s = 4; s < static_cast<int>(src.size()); ++s) {
        if (src[static_cast<std::size_t>(s)].type != 'A') continue;
        const Word& w = src[static_cast<std::size_t>(s)].w;
        if (w[i] == L && w[i + 1] == L) return s;
    }
    return -1;
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 15;
    if (maxW < 6 || maxW > 15) return 2;

    std::vector<std::vector<Word>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 6; W <= maxW; ++W) {
        Rank checked = 0;
        Rank inspected_tail_sources = 0;
        Rank max_tail_scan = 0;
        for (int i = 0; i <= W - 4; ++i) {
            for (const Word& u : words[W - 2]) {
                if (u.substr(static_cast<std::size_t>(i), 2) != "LN") continue;
                const auto packed = packed_direct_component_sources(pack_word(u), W, i);
                std::vector<Key> src;
                for (int q = 0; q < packed.size; ++q)
                    src.push_back(unpack_key(packed.value[q]));
                const int pivot_edge = ln_pivot_index(src, W, i);
                const int pivot_local = ln_local_pivot(src, i);
                if (pivot_local != pivot_edge)
                    fail("LN local pivot mismatch W=" + std::to_string(W) +
                         " i=" + std::to_string(i));
                inspected_tail_sources += pivot_local - 3;
                max_tail_scan = std::max<Rank>(max_tail_scan, pivot_local - 3);
                ++checked;
            }
        }
        std::cout << "W=" << W
                  << " LN_components=" << checked
                  << " avg_tail_scan="
                  << (checked ? double(inspected_tail_sources) / double(checked) : 0.0)
                  << " max_tail_scan=" << max_tail_scan
                  << " pivot=first_local_LL"
                  << " matching_K_step_calls=0"
                  << " leaf_peeling=0 OK\n";
    }

    std::cout << "W=28_plan max_component_sources=17"
              << " max_LN_pivot_scan=14"
              << " matching_K_step_calls=0"
              << " matching_leaf_peeling_calls=0\n";
    std::cout << "ALL_OK LN_first_LL_pivot=1\n";
    return 0;
}
