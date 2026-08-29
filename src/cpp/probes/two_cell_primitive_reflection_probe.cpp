#pragma push_macro("main")
#undef main
#define main two_cell_primitive_meta_probe_main_unused
#include "two_cell_primitive_meta_probe.cpp"
#pragma pop_macro("main")

#include "../../common/two_cell_primitive_reflection.hpp"

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 15;
    if (maxW < 1 || maxW > 15) return 2;

    const auto rt = oneesan::twocell::make_rank_tables();
    Rank primitive_checks = 0;
    Rank word_checks = 0;

    for (int occupied = 1; occupied <= 27; occupied += 2) {
        const Rank pc = rt.primitive[occupied][1];
        if (pc >= (Rank(1) << oneesan::twocell::kPrimitiveMirrorRankBits))
            fail("primitive reflection rank bit budget");
        const std::uint32_t support = oneesan::twocell::low_mask(occupied);
        for (Rank r = 0; r < pc; ++r) {
            const std::uint32_t left = oneesan::twocell::primitive_left_unrank(
                support, occupied, occupied, r, rt);
            const std::uint32_t meta =
                oneesan::twocell::make_primitive_reflection_meta(
                    left, occupied, rt);
            if (meta == 0xffffffffu)
                fail("primitive reflection invalid metadata");
            const Rank mr = oneesan::twocell::primitive_reflection_mirror_rank(meta);
            if (mr >= pc) fail("primitive reflection mirror rank range");

            const std::uint32_t mleft = oneesan::twocell::primitive_left_unrank(
                support, occupied, occupied, mr, rt);
            const std::uint32_t mmeta =
                oneesan::twocell::make_primitive_reflection_meta(
                    mleft, occupied, rt);
            if (oneesan::twocell::primitive_reflection_mirror_rank(mmeta) != r)
                fail("primitive reflection involution rank");

            const oneesan::twocell::PackedWord w{
                support, left, static_cast<std::uint8_t>(occupied)};
            const auto reflected = oneesan::twocell::reflect_word_with_reflection_meta(
                w, meta);
            const auto expected = oneesan::twocell::reflect_packed_word(w);
            if (reflected.support != expected.support ||
                reflected.left != expected.left || reflected.len != expected.len)
                fail("primitive reflection compact word mismatch");
            const Rank rr = oneesan::twocell::primitive_rank(
                reflected.support, reflected.left, occupied, rt);
            if (rr != mr) fail("primitive reflection rank mismatch");
            ++primitive_checks;
        }
        std::cout << "occupied=" << occupied
                  << " primitive=" << pc
                  << " mirror_rank_involution=OK\n";
    }

    for (int W = 1; W <= maxW; ++W) {
        for (const Word& word : gen_words(W)) {
            const auto packed = meta_word(word);
            const int occupied = oneesan::twocell::popcount32(packed.support);
            const Rank r = oneesan::twocell::primitive_rank(
                packed.support, packed.left, packed.len, rt);
            const std::uint32_t compact = oneesan::twocell::primitive_left_unrank(
                oneesan::twocell::low_mask(occupied), occupied, occupied, r, rt);
            const std::uint32_t meta =
                oneesan::twocell::make_primitive_reflection_meta(
                    compact, occupied, rt);
            const auto reflected = oneesan::twocell::reflect_word_with_reflection_meta(
                packed, meta);
            const auto expected = oneesan::twocell::reflect_packed_word(packed);
            if (reflected.support != expected.support ||
                reflected.left != expected.left || reflected.len != expected.len)
                fail("primitive reflection physical word mismatch W=" +
                     std::to_string(W));
            const Rank mr = oneesan::twocell::primitive_reflection_mirror_rank(meta);
            if (oneesan::twocell::primitive_rank(
                    reflected.support, reflected.left, reflected.len, rt) != mr)
                fail("primitive reflection physical mirror rank W=" +
                     std::to_string(W));
            ++word_checks;
        }
    }

    Rank entries = 0;
    for (int occupied = 1; occupied <= 27; occupied += 2)
        entries += rt.primitive[occupied][1];
    if (entries != 3707851ULL) fail("primitive reflection entry total");

    std::cout << "W=28_theory reflection_entries=" << entries
              << " reflection_LUT_MiB="
              << double(entries * sizeof(std::uint32_t)) / double(1ULL << 20)
              << " root_scan_per_reflection=0"
              << " mirror_rank_scan_after_reflection=0\n";
    std::cout << "checks primitive=" << primitive_checks
              << " words=" << word_checks << '\n';
    std::cout << "ALL_OK primitive_reflection_metadata=1\n";
    return 0;
}
