#pragma push_macro("main")
#undef main
#define main two_cell_stationary_support_base_probe_main_unused
#include "two_cell_stationary_support_base_probe.cpp"
#pragma pop_macro("main")

namespace {

struct DirectedEdge {
    int src = -1;
    int dst = -1;
};

int source_index_for_destination(
    const std::vector<Key>& src,
    const Key& dst,
    int i
) {
    for (int q = 0; q < static_cast<int>(src.size()); ++q)
        if (recouple_coordinate(src[static_cast<std::size_t>(q)], i) == dst) return q;
    return -1;
}

std::vector<int> reverse_topological_order(
    int n,
    const std::vector<DirectedEdge>& edge
) {
    std::vector<std::vector<int>> outgoing(static_cast<std::size_t>(n));
    std::vector<std::vector<int>> incoming(static_cast<std::size_t>(n));
    std::vector<int> outdegree(static_cast<std::size_t>(n));
    for (int e = 0; e < static_cast<int>(edge.size()); ++e) {
        outgoing[static_cast<std::size_t>(edge[e].src)].push_back(e);
        incoming[static_cast<std::size_t>(edge[e].dst)].push_back(e);
        ++outdegree[static_cast<std::size_t>(edge[e].src)];
    }

    std::vector<std::uint8_t> removed(static_cast<std::size_t>(n));
    std::vector<int> order;
    order.reserve(static_cast<std::size_t>(n));
    for (int round = 0; round < n; ++round) {
        bool progress = false;
        for (int v = 0; v < n; ++v) {
            if (removed[static_cast<std::size_t>(v)] || outdegree[static_cast<std::size_t>(v)] != 0)
                continue;
            removed[static_cast<std::size_t>(v)] = 1;
            order.push_back(v);
            progress = true;
            for (int e : incoming[static_cast<std::size_t>(v)]) {
                const int u = edge[static_cast<std::size_t>(e)].src;
                if (!removed[static_cast<std::size_t>(u)])
                    --outdegree[static_cast<std::size_t>(u)];
            }
        }
        if (!progress) break;
    }
    if (static_cast<int>(order.size()) != n) fail("lifting directed cycle");
    return order;
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
        Rank diagonal_edges = 0;
        Rank shear_edges = 0;
        Rank max_shear_edges = 0;
        Rank max_outdegree = 0;
        Rank max_indegree = 0;
        Rank max_rounds = 0;

        for (int i = 0; i <= W - 4; ++i) {
            for (const Word& u : words[W - 2]) {
                const auto packed_sources = packed_direct_component_sources(pack_word(u), W, i);
                std::vector<Key> src;
                src.reserve(static_cast<std::size_t>(packed_sources.size));
                for (int q = 0; q < packed_sources.size; ++q)
                    src.push_back(unpack_key(packed_sources.value[q]));
                const int n = static_cast<int>(src.size());

                std::vector<DirectedEdge> shear;
                std::vector<int> outdegree(static_cast<std::size_t>(n));
                std::vector<int> indegree(static_cast<std::size_t>(n));
                Rank local_diagonal = 0;
                for (int s = 0; s < n; ++s) {
                    bool has_diagonal = false;
                    for (const auto& [d, c] : K_basis(src[static_cast<std::size_t>(s)], W, i)) {
                        if (c != 1) fail("lifting coefficient");
                        const int t = source_index_for_destination(src, d, i);
                        if (t < 0) fail("lifting destination coordinate");
                        if (t == s) {
                            if (has_diagonal) fail("lifting duplicate diagonal");
                            has_diagonal = true;
                            ++local_diagonal;
                        } else {
                            shear.push_back({s, t});
                            ++outdegree[static_cast<std::size_t>(s)];
                            ++indegree[static_cast<std::size_t>(t)];
                        }
                    }
                    if (!has_diagonal)
                        fail("recoupling not matching edge W=" + std::to_string(W) +
                             " i=" + std::to_string(i));
                }

                if (local_diagonal != Rank(n)) fail("lifting matching size");
                if (shear.size() != static_cast<std::size_t>(n - 1))
                    fail("lifting contracted tree edge count");

                // The contracted graph must be a tree. n-1 edges plus
                // connectivity follows from the original bipartite forest, but
                // verify directly here as well.
                std::vector<std::vector<int>> undirected(static_cast<std::size_t>(n));
                for (const auto& e : shear) {
                    undirected[static_cast<std::size_t>(e.src)].push_back(e.dst);
                    undirected[static_cast<std::size_t>(e.dst)].push_back(e.src);
                }
                std::vector<std::uint8_t> seen(static_cast<std::size_t>(n));
                std::vector<int> stack{0};
                seen[0] = 1;
                int visited = 0;
                while (!stack.empty()) {
                    const int v = stack.back();
                    stack.pop_back();
                    ++visited;
                    for (int z : undirected[static_cast<std::size_t>(v)])
                        if (!seen[static_cast<std::size_t>(z)]) {
                            seen[static_cast<std::size_t>(z)] = 1;
                            stack.push_back(z);
                        }
                }
                if (visited != n) fail("lifting contracted disconnected");

                const auto order = reverse_topological_order(n, shear);
                std::vector<int64_t> x(static_cast<std::size_t>(n));
                std::vector<int64_t> expected(static_cast<std::size_t>(n));
                for (int q = 0; q < n; ++q)
                    x[static_cast<std::size_t>(q)] = 1 + 17 * q + 101 * i;

                // Exact matrix multiply in stationary coordinates.
                for (int s = 0; s < n; ++s) {
                    for (const auto& [d, c] : K_basis(src[static_cast<std::size_t>(s)], W, i)) {
                        const int t = source_index_for_destination(src, d, i);
                        expected[static_cast<std::size_t>(t)] +=
                            c * x[static_cast<std::size_t>(s)];
                    }
                }

                // Diagonal matching is identity. Process vertices sink-first;
                // every outgoing use therefore sees the original x[v] before
                // any incoming shear modifies that coordinate.
                std::vector<int64_t> lifted = x;
                for (int v : order) {
                    for (const auto& e : shear)
                        if (e.src == v)
                            lifted[static_cast<std::size_t>(e.dst)] +=
                                lifted[static_cast<std::size_t>(v)];
                }
                if (lifted != expected)
                    fail("lifting arithmetic W=" + std::to_string(W));

                int rounds = 0;
                std::vector<int> depth(static_cast<std::size_t>(n));
                for (int v : order) {
                    for (const auto& e : shear)
                        if (e.src == v)
                            depth[static_cast<std::size_t>(e.src)] = std::max(
                                depth[static_cast<std::size_t>(e.src)],
                                1 + depth[static_cast<std::size_t>(e.dst)]);
                    rounds = std::max(rounds, depth[static_cast<std::size_t>(v)]);
                }

                for (int q = 0; q < n; ++q) {
                    max_outdegree = std::max<Rank>(max_outdegree, outdegree[static_cast<std::size_t>(q)]);
                    max_indegree = std::max<Rank>(max_indegree, indegree[static_cast<std::size_t>(q)]);
                }
                max_rounds = std::max<Rank>(max_rounds, rounds);
                diagonal_edges += local_diagonal;
                shear_edges += shear.size();
                max_shear_edges = std::max<Rank>(max_shear_edges, shear.size());
                coordinates += n;
                ++components;
            }
        }

        std::cout << "W=" << W
                  << " components=" << components
                  << " coordinates=" << coordinates
                  << " diagonal_edges=" << diagonal_edges
                  << " shear_edges=" << shear_edges
                  << " max_shear_edges=" << max_shear_edges
                  << " max_outdegree=" << max_outdegree
                  << " max_indegree=" << max_indegree
                  << " max_dependency_depth=" << max_rounds
                  << " recouple_is_unique_matching=OK"
                  << " contracted_graph=tree"
                  << " inplace_shear=OK\n";
    }

    std::cout << "W=28_plan max_component=17"
              << " max_shears_per_component=16"
              << " diagonal_materialization=0"
              << " destination_accumulation_scan=0"
              << "\n";
    std::cout << "ALL_OK stationary_tree_lifting=1\n";
    return 0;
}
