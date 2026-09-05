#pragma push_macro("main")
#undef main
#define main two_cell_stationary_layout_probe_main_unused
#include "two_cell_stationary_layout_probe.cpp"
#pragma pop_macro("main")

#include "../../common/two_cell_stationary_rank.hpp"

namespace {

oneesan::twocell::PackedKey stationary_pack(const Key& k) {
    oneesan::twocell::PackedKey z{};
    z.type = static_cast<std::uint8_t>(k.type == 'C');
    for (int p = 0; p < static_cast<int>(k.w.size()); ++p) {
        const std::uint32_t bit = std::uint32_t(1) << p;
        if (k.w[p] != N) z.support |= bit;
        if (k.w[p] == L) z.left |= bit;
    }
    return z;
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 13;
    if (maxW < 5 || maxW > 15) return 2;

    const auto rt = oneesan::twocell::make_rank_tables();
    const auto st = oneesan::twocell::make_stationary_rank_tables(rt);
    std::vector<std::vector<Word>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 5; W <= maxW; ++W) {
        const Rank expected = Rank(words[W - 1].size()) +
                              Rank(words[W - 2].size()) - Rank(words[W - 3].size());
        if (st.total[W] != expected || st.a_total[W] != words[W - 1].size())
            fail("stationary rank dimension W=" + std::to_string(W));

        Rank basis_checks = 0;
        Rank recoupling_checks = 0;
        for (int active = 0; active <= W - 3; ++active) {
            std::vector<std::uint8_t> seen(static_cast<std::size_t>(expected));
            for (const Key& k : q_basis(W, active, words)) {
                const auto p = stationary_pack(k);
                const Rank r = oneesan::twocell::stationary_rank(p, W, active, rt, st);
                if (r >= expected || seen[static_cast<std::size_t>(r)]++)
                    fail("stationary rank collision W=" + std::to_string(W));

                const Rank primitive = oneesan::twocell::primitive_rank(
                    p.support, p.left, static_cast<int>(k.w.size()), rt);
                const Rank r2 = oneesan::twocell::stationary_rank_with_primitive(
                    p, W, active, primitive, rt, st);
                if (r != r2) fail("stationary precomputed primitive rank");
                ++basis_checks;
            }
            for (std::uint8_t x : seen)
                if (x != 1) fail("stationary rank gap W=" + std::to_string(W));
        }

        for (int i = 0; i <= W - 4; ++i) {
            for (const Word& u : words[W - 2]) {
                const auto sources = packed_direct_component_sources(pack_word(u), W, i);
                for (int q = 0; q < sources.size; ++q) {
                    const Key src = unpack_key(sources.value[q]);
                    const Key dst = recouple_coordinate(src, i);
                    const auto ps = stationary_pack(src);
                    const auto pd = stationary_pack(dst);
                    const Rank primitive = oneesan::twocell::primitive_rank(
                        ps.support, ps.left, static_cast<int>(src.w.size()), rt);
                    const Rank sr = oneesan::twocell::stationary_rank_with_primitive(
                        ps, W, i, primitive, rt, st);
                    const Rank dr = oneesan::twocell::stationary_rank_with_primitive(
                        pd, W, i + 1, primitive, rt, st);
                    if (sr != dr)
                        fail("stationary recoupling rank identity W=" + std::to_string(W));
                    ++recoupling_checks;
                }
            }
        }

        std::cout << "W=" << W
                  << " states=" << expected
                  << " A=" << st.a_total[W]
                  << " C=" << (st.total[W] - st.a_total[W])
                  << " basis_checks=" << basis_checks
                  << " recoupling_checks=" << recoupling_checks
                  << " fixed_rank_bijection=OK recoupling_same_address=OK\n";
    }

    const Rank dim28 = st.total[28];
    if (dim28 != 165727043758ULL) fail("stationary W28 dimension");
    std::cout << "W=28_theory states=" << dim28
              << " one_u32_vector_GiB="
              << double(dim28 * 4ULL) / double(1ULL << 30)
              << " rank_table_KiB="
              << double(sizeof(rt) + sizeof(st)) / 1024.0
              << " destination_vector_required=0"
              << "\n";
    std::cout << "ALL_OK stationary_reduced_rank=1\n";
    return 0;
}
