#pragma push_macro("main")
#undef main
#define main two_cell_reverse_channel_probe_main_unused
#include "two_cell_reverse_channel_probe.cpp"
#pragma pop_macro("main")

#include "../../common/two_cell_stationary_rank.hpp"

namespace {

oneesan::twocell::PackedKey reverse_stationary_pack(const Key& k) {
    oneesan::twocell::PackedKey z{};
    z.type = static_cast<std::uint8_t>(k.type == 'C');
    for (int p = 0; p < static_cast<int>(k.w.size()); ++p) {
        const std::uint32_t bit = std::uint32_t(1) << p;
        if (k.w[p] != N) z.support |= bit;
        if (k.w[p] == L) z.left |= bit;
    }
    return z;
}

Key reverse_recouple_coordinate(Key src, int pair) {
    if (src.type == 'A') return src;
    if (src.type != 'C') fail("reverse stationary coordinate type");
    const int active = pair - 1;
    const int next_active = active - 1;
    if (next_active >= 0 && src.w[next_active] == N) {
        if (src.w[active] == N) fail("reverse stationary active vacancy");
        std::swap(src.w[next_active], src.w[active]);
    }
    return src;
}

bool vec_has_unit_edge(const CVec& v, const Key& dst) {
    const auto it = v.find(dst);
    return it != v.end() && it->second == 1;
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
        Rank source_checks = 0;
        Rank same_address_checks = 0;
        Rank diagonal_edge_checks = 0;
        Rank c_swaps = 0;

        for (int i = 0; i <= W - 4; ++i) {
            const int pair = W - 2 - i;
            if (pair < 2 || pair > W - 2) continue;
            const int src_active = pair - 1;
            const int dst_active = pair - 2;

            const auto rsrc = reverse_q_basis(W, pair, words);
            std::set<Key> rsrc_set(rsrc.begin(), rsrc.end());
            const auto rdst = reverse_q_basis(W, pair - 1, words);
            std::set<Key> rdst_set(rdst.begin(), rdst.end());

            for (const Key& src : rsrc) {
                const Key dst = reverse_recouple_coordinate(src, pair);
                if (!rdst_set.count(dst))
                    fail("reverse stationary recoupled destination W=" + std::to_string(W));

                const auto ps = reverse_stationary_pack(src);
                const auto pd = reverse_stationary_pack(dst);
                const int len = static_cast<int>(src.w.size());
                const Rank primitive = oneesan::twocell::primitive_rank(
                    ps.support, ps.left, len, rt);
                const Rank sr = oneesan::twocell::stationary_rank_with_primitive(
                    ps, W, src_active, primitive, rt, st);
                const Rank dr = oneesan::twocell::stationary_rank_with_primitive(
                    pd, W, dst_active, primitive, rt, st);
                if (sr != dr)
                    fail("reverse stationary same address W=" + std::to_string(W) +
                         " pair=" + std::to_string(pair));

                const CVec edges = K_reverse_basis(src, W, pair);
                if (!vec_has_unit_edge(edges, dst))
                    fail("reverse stationary diagonal edge W=" + std::to_string(W) +
                         " pair=" + std::to_string(pair));

                if (src.type == 'C' && !(src == dst)) {
                    if (src.w[src_active] == N || src.w[dst_active] != N)
                        fail("reverse stationary bad C swap source");
                    if (dst.w[dst_active] == N || dst.w[src_active] != N)
                        fail("reverse stationary bad C swap destination");
                    ++c_swaps;
                }
                ++source_checks;
                ++same_address_checks;
                ++diagonal_edge_checks;
            }

            // Reflection of every forward component must give one reverse
            // component whose stationary source and destination coordinate sets
            // are identical. This also checks that component boundaries survive
            // the fixed-basis normalization rather than merely individual edges.
            for (const Word& u : words[W - 2]) {
                const Key fseed = project_key(Key{'C', u}, i, W);
                std::set<Key> fsources{fseed};
                std::deque<Key> queue{fseed};
                while (!queue.empty()) {
                    const Key s = queue.front();
                    queue.pop_front();
                    for (const auto& [d, c] : K_basis(s, W, i)) {
                        if (c != 1) fail("reverse stationary forward nonunit");
                        for (const Key& p : inverse_K(d, W, i))
                            if (fsources.insert(p).second) queue.push_back(p);
                    }
                }

                std::set<Rank> src_ranks, dst_ranks;
                for (const Key& fsrc : fsources) {
                    const Key rs = reflect_key(fsrc);
                    if (!rsrc_set.count(rs))
                        fail("reverse stationary reflected source layout");
                    const Key rd = reverse_recouple_coordinate(rs, pair);
                    const auto prs = reverse_stationary_pack(rs);
                    const auto prd = reverse_stationary_pack(rd);
                    const Rank primitive = oneesan::twocell::primitive_rank(
                        prs.support, prs.left, static_cast<int>(rs.w.size()), rt);
                    src_ranks.insert(oneesan::twocell::stationary_rank_with_primitive(
                        prs, W, src_active, primitive, rt, st));
                    dst_ranks.insert(oneesan::twocell::stationary_rank_with_primitive(
                        prd, W, dst_active, primitive, rt, st));
                }
                if (src_ranks != dst_ranks)
                    fail("reverse stationary component coordinate set W=" +
                         std::to_string(W));
            }
        }

        std::cout << "W=" << W
                  << " reverse_sources=" << source_checks
                  << " same_address=" << same_address_checks
                  << " unit_matching_edges=" << diagonal_edge_checks
                  << " C_left_swaps=" << c_swaps
                  << " stationary_component_sets=OK"
                  << " reverse_single_buffer=OK\n";
    }

    std::cout << "W=28_theory reverse_stationary_vector_GiB="
              << double(st.total[28] * 4ULL) / double(1ULL << 30)
              << " reverse_destination_vector_required=0"
              << " reverse_destination_rank_required=0\n";
    std::cout << "ALL_OK reverse_stationary_two_cell=1\n";
    return 0;
}
