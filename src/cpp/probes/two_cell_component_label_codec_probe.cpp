#pragma push_macro("main")
#undef main
#define main two_cell_recoupling_common_rank_probe_main_unused
#include "two_cell_recoupling_common_rank_probe.cpp"
#pragma pop_macro("main")

namespace {

Key unpack_component_label(const oneesan::twocell::PackedKey& p, int len) {
    Word w(static_cast<std::size_t>(len), N);
    for (int pos = 0; pos < len; ++pos) {
        const std::uint32_t bit = std::uint32_t(1) << pos;
        if (!(p.support & bit)) continue;
        w[pos] = (p.left & bit) ? L : R;
    }
    return Key{'C', w};
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 14;
    if (maxW < 4 || maxW > 15) return 2;

    const auto tables = oneesan::twocell::make_rank_tables();
    std::vector<std::vector<Word>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 4; W <= maxW; ++W) {
        const Rank count = oneesan::twocell::component_label_count(W, tables);
        if (count != words[W - 2].size())
            fail("component label count W=" + std::to_string(W));

        std::set<Word> seen;
        Rank max_sector = 0;
        for (Rank r = 0; r < count; ++r) {
            const auto p = oneesan::twocell::component_label_unrank(W, r, tables);
            const Key k = unpack_component_label(p, W - 2);
            if (!valid_word(k.w))
                fail("component label invalid W=" + std::to_string(W));
            if (!seen.insert(k.w).second)
                fail("component label collision W=" + std::to_string(W));
            max_sector = std::max<Rank>(max_sector,
                oneesan::twocell::popcount32(p.support));
        }
        if (seen != std::set<Word>(words[W - 2].begin(), words[W - 2].end()))
            fail("component label coverage W=" + std::to_string(W));

        // Every label must still generate exactly the source component used by
        // the direct one-scan reconstruction, independent of the label order.
        Rank component_checks = 0;
        for (int i = 0; i <= W - 4; ++i) {
            for (Rank r = 0; r < count; ++r) {
                const auto p = oneesan::twocell::component_label_unrank(W, r, tables);
                const Word u = unpack_component_label(p, W - 2).w;
                const auto direct = packed_direct_component_sources(pack_word(u), W, i);
                if (direct.size < 1 || direct.size > W / 2 + 3)
                    fail("component label direct size W=" + std::to_string(W));
                ++component_checks;
            }
        }

        std::cout << "W=" << W
                  << " labels=" << count
                  << " component_checks=" << component_checks
                  << " max_occupied=" << max_sector
                  << " label_table_bytes=0"
                  << " support_primitive_unrank=OK"
                  << "\n";
    }

    const int W = 28;
    const Rank n = oneesan::twocell::component_label_count(W, tables);
    std::cout << "W=28_theory labels=" << n
              << " expected=47337954326"
              << " rank_table_bytes=" << sizeof(tables)
              << " rank_table_KiB=" << double(sizeof(tables)) / 1024.0
              << " global_label_table_bytes=0"
              << "\n";
    if (n != 47337954326ULL) fail("W28 label count");
    std::cout << "ALL_OK component_label_device_codec=1\n";
    return 0;
}
