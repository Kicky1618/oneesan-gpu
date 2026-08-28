#pragma push_macro("main")
#undef main
#define main two_cell_primitive_rank_reuse_probe_main_unused
#include "two_cell_primitive_rank_reuse_probe.cpp"
#pragma pop_macro("main")

namespace {

Key stationary_key(Key k, int active) {
    if (k.type == 'A') return k;
    if (k.type != 'C' || k.w[active] == N) fail("stationary C source");
    for (int p = active - 1; p >= 0; --p) {
        if (k.w[p] == N) {
            if (k.w[p + 1] == N) fail("stationary marker lost");
            std::swap(k.w[p], k.w[p + 1]);
        }
        // If both positions are occupied, only the distinguished active strand
        // changes; the physical word is already in canonical order.
    }
    if (k.w[0] == N) fail("stationary canonical bit0");
    return k;
}

std::uint32_t support_of(const Word& w) {
    std::uint32_t z = 0;
    for (int p = 0; p < static_cast<int>(w.size()); ++p)
        if (w[p] != N) z |= std::uint32_t(1) << p;
    return z;
}

std::uint32_t stationary_support_formula(std::uint32_t support, int active) {
    const std::uint32_t prefix = oneesan::twocell::low_mask(active);
    const std::uint32_t suffix = ~oneesan::twocell::low_mask(active + 1);
    return (support & suffix) | 1u | ((support & prefix) << 1);
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 13;
    if (maxW < 5 || maxW > 15) return 2;

    std::vector<std::vector<Word>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 5; W <= maxW; ++W) {
        std::set<Key> fixed_basis;
        for (const Word& w : words[W - 1]) fixed_basis.insert(Key{'A', w});
        for (const Word& w : words[W - 2])
            if (w[0] != N) fixed_basis.insert(Key{'C', w});

        Rank basis_checks = 0;
        Rank component_checks = 0;
        Rank coordinate_checks = 0;
        for (int active = 0; active <= W - 3; ++active) {
            const auto q = q_basis(W, active, words);
            std::set<Key> mapped;
            for (const Key& k : q) {
                const Key z = stationary_key(k, active);
                if (!mapped.insert(z).second) fail("stationary basis collision");
                if (k.type == 'C') {
                    const std::uint32_t formula = stationary_support_formula(
                        support_of(k.w), active);
                    if (support_of(z.w) != formula)
                        fail("stationary support formula");
                    if (primitive_signature(k) != primitive_signature(z))
                        fail("stationary primitive changed");
                }
                ++basis_checks;
            }
            if (mapped != fixed_basis)
                fail("stationary basis coverage W=" + std::to_string(W) +
                     " active=" + std::to_string(active));
        }

        for (int i = 0; i <= W - 4; ++i) {
            for (const Word& u : words[W - 2]) {
                const auto packed_sources = packed_direct_component_sources(pack_word(u), W, i);
                std::set<Key> source_stationary;
                std::set<Key> destination_stationary;
                for (int q = 0; q < packed_sources.size; ++q) {
                    const Key s = unpack_key(packed_sources.value[q]);
                    const Key fs = stationary_key(s, i);
                    source_stationary.insert(fs);

                    const Key rd = recouple_coordinate(s, i);
                    const Key fd = stationary_key(rd, i + 1);
                    if (fs != fd)
                        fail("stationary recoupling identity W=" + std::to_string(W));
                    ++coordinate_checks;

                    for (const auto& [d, c] : K_basis(s, W, i)) {
                        if (c != 1) fail("stationary nonunit K");
                        destination_stationary.insert(stationary_key(d, i + 1));
                    }
                }
                if (source_stationary != destination_stationary)
                    fail("stationary component set mismatch W=" + std::to_string(W));
                ++component_checks;
            }
        }

        const Rank dim = fixed_basis.size();
        const Rank cdim = Rank(words[W - 2].size()) - Rank(words[W - 3].size());
        if (dim != Rank(words[W - 1].size()) + cdim)
            fail("stationary dimension formula");

        std::cout << "W=" << W
                  << " fixed_states=" << dim
                  << " fixed_C=" << cdim
                  << " basis_checks=" << basis_checks
                  << " component_checks=" << component_checks
                  << " coordinate_checks=" << coordinate_checks
                  << " stationary_basis=OK recoupling_address_identity=OK\n";
    }

    const Rank dim28 = 165727043758ULL;
    const double one_gib = double(dim28) * 4.0 / double(1ULL << 30);
    std::cout << "W=28_theory fixed_states=" << dim28
              << " one_u32_vector_GiB=" << one_gib
              << " two_vector_GiB=" << (2.0 * one_gib)
              << " destination_vector_bytes_saved=" << (dim28 * 4ULL)
              << "\n";
    std::cout << "ALL_OK stationary_reduced_layout=1\n";
    return 0;
}
