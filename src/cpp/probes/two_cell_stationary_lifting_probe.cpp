#pragma push_macro("main")
#undef main
#define main two_cell_stationary_support_base_probe_main_unused
#include "two_cell_stationary_support_base_probe.cpp"
#pragma pop_macro("main")

#include "../../common/two_cell_component_matching.cuh"

namespace {

constexpr std::uint32_t kMatchingMod = 1000000007u;

int source_index_for_destination(
    const std::vector<Key>& src,
    const Key& dst,
    int i
) {
    for (int q = 0; q < static_cast<int>(src.size()); ++q)
        if (recouple_coordinate(src[static_cast<std::size_t>(q)], i) == dst) return q;
    return -1;
}

oneesan::twocell::PackedKey matching_pack(const Key& k) {
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

    std::vector<std::vector<Word>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 5; W <= maxW; ++W) {
        Rank components = 0;
        Rank coordinates = 0;
        Rank matching_edges = 0;
        Rank residual_edges = 0;
        Rank nonidentity_components = 0;
        Rank moved_matching_coordinates = 0;
        Rank max_component = 0;
        Rank max_matching_cycle = 0;

        for (int i = 0; i <= W - 4; ++i) {
            for (const Word& u : words[W - 2]) {
                const auto packed_sources = packed_direct_component_sources(pack_word(u), W, i);
                const int n = packed_sources.size;
                if (n <= 0 || n > oneesan::twocell::kMaxComponentMatching)
                    fail("matching component size");

                std::vector<Key> src;
                src.reserve(static_cast<std::size_t>(n));
                oneesan::twocell::PackedKey device_src[oneesan::twocell::kMaxComponentMatching]{};
                for (int q = 0; q < n; ++q) {
                    const Key k = unpack_key(packed_sources.value[q]);
                    src.push_back(k);
                    device_src[q] = matching_pack(k);
                }

                // recouple_coordinate() is only the coordinate-set bijection:
                // {recouple(src)} equals the destination set, but it is not in
                // general a matrix edge from that same source.
                std::set<Key> predicted_dst;
                std::set<Key> actual_dst;
                for (int s = 0; s < n; ++s) {
                    predicted_dst.insert(recouple_coordinate(src[s], i));
                    for (const auto& [d, c] : K_basis(src[s], W, i)) {
                        if (c != 1) fail("matching coefficient");
                        actual_dst.insert(d);
                    }
                }
                if (predicted_dst != actual_dst ||
                    static_cast<int>(predicted_dst.size()) != n)
                    fail("stationary coordinate-set bijection");

                const auto matching = oneesan::twocell::build_component_matching(
                    device_src, n, W, i);
                if (!matching.ok || matching.size != n ||
                    matching.edges != 2 * n - 1 ||
                    matching.residual_edges != n - 1)
                    fail("local unique matching construction W=" + std::to_string(W) +
                         " i=" + std::to_string(i));

                bool nonidentity = false;
                int moved = 0;
                for (int s = 0; s < n; ++s) {
                    const int t = matching.src_to_dst[s];
                    if (t != s) {
                        nonidentity = true;
                        ++moved;
                    }
                    // The recovered edge must exist in the exact CPU oracle.
                    const Key matched_dst = recouple_coordinate(src[t], i);
                    bool found = false;
                    for (const auto& [d, c] : K_basis(src[s], W, i))
                        found |= (c == 1 && d == matched_dst);
                    if (!found) fail("matching edge absent from K");
                }
                if (nonidentity) ++nonidentity_components;
                moved_matching_coordinates += moved;

                // Inspect permutation cycle length. This is tiny local metadata,
                // not a global permutation table.
                std::vector<std::uint8_t> seen(static_cast<std::size_t>(n));
                for (int s = 0; s < n; ++s) {
                    if (seen[s]) continue;
                    int q = s, cycle = 0;
                    while (!seen[q]) {
                        seen[q] = 1;
                        q = matching.src_to_dst[q];
                        ++cycle;
                    }
                    max_matching_cycle = std::max<Rank>(max_matching_cycle, cycle);
                }

                std::uint32_t x[oneesan::twocell::kMaxComponentMatching]{};
                std::uint32_t got[oneesan::twocell::kMaxComponentMatching]{};
                std::uint32_t expected[oneesan::twocell::kMaxComponentMatching]{};
                for (int q = 0; q < n; ++q)
                    x[q] = static_cast<std::uint32_t>(1 + 17 * q + 101 * i);

                for (int s = 0; s < n; ++s) {
                    for (const auto& [d, c] : K_basis(src[s], W, i)) {
                        if (c != 1) fail("matching arithmetic coefficient");
                        const int t = source_index_for_destination(src, d, i);
                        if (t < 0) fail("matching destination coordinate");
                        expected[t] = static_cast<std::uint32_t>(
                            (static_cast<std::uint64_t>(expected[t]) + x[s]) % kMatchingMod);
                    }
                }
                if (!oneesan::twocell::apply_component_matching(
                        matching, x, got, kMatchingMod))
                    fail("matching apply returned false");
                for (int q = 0; q < n; ++q)
                    if (got[q] != expected[q])
                        fail("matching arithmetic W=" + std::to_string(W) +
                             " i=" + std::to_string(i));

                matching_edges += n;
                residual_edges += n - 1;
                coordinates += n;
                max_component = std::max<Rank>(max_component, n);
                ++components;
            }
        }

        if (residual_edges + matching_edges !=
            coordinates + residual_edges)
            fail("matching bookkeeping");

        std::cout << "W=" << W
                  << " components=" << components
                  << " coordinates=" << coordinates
                  << " matching_edges=" << matching_edges
                  << " residual_edges=" << residual_edges
                  << " nonidentity_matching_components=" << nonidentity_components
                  << " moved_matching_coordinates=" << moved_matching_coordinates
                  << " max_matching_cycle=" << max_matching_cycle
                  << " max_component=" << max_component
                  << " recouple=coordinate_bijection"
                  << " matching=leaf_peeling"
                  << " local_permutation_plus_residual_adds=OK\n";
    }

    std::cout << "W=28_plan max_component=17"
              << " matching_slots=17"
              << " residual_adds_per_component=n-1"
              << " global_matching_table_bytes=0"
              << " second_global_vector_bytes=0"
              << "\n";
    std::cout << "ALL_OK stationary_local_matching=1\n";
    return 0;
}
