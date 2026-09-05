#pragma push_macro("main")
#undef main
#define main two_cell_component_device_probe_main_unused
#include "two_cell_component_device_probe.cpp"
#pragma pop_macro("main")

#include "../../common/two_cell_primitive_meta.hpp"

namespace {

oneesan::twocell::PackedWord meta_word(const Word& w) {
    oneesan::twocell::PackedWord z{};
    z.len = static_cast<std::uint8_t>(w.size());
    for (int p = 0; p < static_cast<int>(w.size()); ++p) {
        const std::uint32_t bit = std::uint32_t(1) << p;
        if (w[p] != N) z.support |= bit;
        if (w[p] == L) z.left |= bit;
    }
    return z;
}

std::uint32_t compact_left_from_word(oneesan::twocell::PackedWord w) {
    std::uint32_t compact = 0;
    int ordinal = 0;
    for (int p = 0; p < w.len; ++p) {
        const std::uint32_t bit = std::uint32_t(1) << p;
        if (!(w.support & bit)) continue;
        if (w.left & bit) compact |= std::uint32_t(1) << ordinal;
        ++ordinal;
    }
    return compact;
}

bool same_meta_word(
    oneesan::twocell::PackedWord a,
    oneesan::twocell::PackedWord b
) {
    return a.support == b.support && a.left == b.left && a.len == b.len;
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 15;
    if (maxW < 1 || maxW > 15) return 2;

    const auto rt = oneesan::twocell::make_rank_tables();
    Rank word_checks = 0;
    Rank select_checks = 0;

    for (int W = 1; W <= maxW; ++W) {
        const auto words = gen_words(W);
        for (const Word& w : words) {
            const auto packed = meta_word(w);
            const int occupied = oneesan::twocell::popcount32(packed.support);
            const Rank primitive = oneesan::twocell::primitive_rank(
                packed.support, packed.left, packed.len, rt);
            const std::uint32_t compact_support =
                oneesan::twocell::low_mask(occupied);
            const std::uint32_t compact_left =
                oneesan::twocell::primitive_left_unrank(
                    compact_support, occupied, occupied, primitive, rt);
            if (compact_left != compact_left_from_word(packed))
                fail("primitive meta compact left mismatch W=" + std::to_string(W));

            const int root_ordinal = oneesan::twocell::compact_root_ordinal(
                compact_left, occupied);
            if (root_ordinal < 0 || root_ordinal >= occupied)
                fail("primitive meta root ordinal range W=" + std::to_string(W));
            const int root_position = oneesan::twocell::select_nth_set32(
                packed.support, root_ordinal);
            if (root_position != oneesan::twocell::root_position(packed))
                fail("primitive meta root position mismatch W=" + std::to_string(W));

            const std::uint32_t meta = oneesan::twocell::pack_primitive_meta(
                compact_left, root_ordinal);
            if (oneesan::twocell::primitive_meta_left(meta) != compact_left ||
                oneesan::twocell::primitive_meta_root_ordinal(meta) != root_ordinal)
                fail("primitive meta pack roundtrip W=" + std::to_string(W));

            const auto got = oneesan::twocell::reflect_word_with_primitive_meta(
                packed, meta);
            const auto expected = oneesan::twocell::reflect_packed_word(packed);
            if (!same_meta_word(got, expected))
                fail("primitive meta reflection mismatch W=" + std::to_string(W));
            ++word_checks;

            for (int q = 0; q < occupied; ++q) {
                const int p = oneesan::twocell::select_nth_set32(packed.support, q);
                if (p < 0 || !((packed.support >> p) & 1u) ||
                    oneesan::twocell::popcount32(
                        packed.support & oneesan::twocell::low_mask(p)) != q)
                    fail("primitive meta select mismatch W=" + std::to_string(W));
                ++select_checks;
            }
        }
        std::cout << "W=" << W
                  << " words=" << words.size()
                  << " primitive_meta_reflection=OK"
                  << " select_nth_set=OK\n";
    }

    Rank full_entries = 0;
    Rank label_entries = 0;
    for (int occupied = 1; occupied <= 27; occupied += 2) {
        const Rank pc = rt.primitive[occupied][1];
        full_entries += pc;
        if (occupied <= 25) label_entries += pc;
        const std::uint32_t support = oneesan::twocell::low_mask(occupied);
        for (Rank r = 0; r < pc; ++r) {
            const std::uint32_t left = oneesan::twocell::primitive_left_unrank(
                support, occupied, occupied, r, rt);
            const int root = oneesan::twocell::compact_root_ordinal(left, occupied);
            if (root < 0 || root > 26)
                fail("primitive meta full LUT root range");
            const std::uint32_t meta = oneesan::twocell::pack_primitive_meta(left, root);
            if (oneesan::twocell::primitive_meta_left(meta) != left)
                fail("primitive meta full LUT pack");
        }
    }

    if (full_entries != 3707851ULL || label_entries != 1033411ULL)
        fail("primitive meta LUT entry totals");
    std::cout << "W=28_theory full_state_meta_entries=" << full_entries
              << " full_state_meta_MiB="
              << double(full_entries * sizeof(std::uint32_t)) / double(1ULL << 20)
              << " label_meta_entries=" << label_entries
              << " label_meta_MiB="
              << double(label_entries * sizeof(std::uint32_t)) / double(1ULL << 20)
              << " root_extra_bytes=0\n";
    std::cout << "checks words=" << word_checks
              << " select=" << select_checks << '\n';
    std::cout << "ALL_OK primitive_meta_reflection=1\n";
    return 0;
}
