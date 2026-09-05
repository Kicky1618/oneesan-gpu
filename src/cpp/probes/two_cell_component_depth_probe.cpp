#pragma push_macro("main")
#undef main
#define main two_cell_inverse_gather_probe_main_unused
#include "two_cell_inverse_gather_probe.cpp"
#pragma pop_macro("main")

#include <deque>

namespace {

int expected_seed_depth(const Word& u, int fixed) {
    assert(fixed >= 0 && fixed + 1 < static_cast<int>(u.size()));
    if (u[fixed] == N) return 1;
    if (u[fixed + 1] == N) return 7;
    if (u[fixed] == L && u[fixed + 1] == R) return 7;
    return 4;
}

int local_component_depth(const Key& seed, int W, int i) {
    std::map<Key, int> source_distance;
    std::map<Key, int> destination_distance;
    std::deque<Key> queue;
    source_distance.emplace(seed, 0);
    queue.push_back(seed);
    int max_distance = 0;

    while (!queue.empty()) {
        const Key src = queue.front();
        queue.pop_front();
        const int sd = source_distance.at(src);
        for (const auto& [dst, c] : K_basis(src, W, i)) {
            if (c != 1) fail("component depth nonunit");
            auto [dit, inserted_dst] = destination_distance.emplace(dst, sd + 1);
            if (!inserted_dst && dit->second > sd + 1) dit->second = sd + 1;
            max_distance = std::max(max_distance, dit->second);
            for (const Key& pre : inverse_K(dst, W, i)) {
                auto [sit, inserted_src] = source_distance.emplace(pre, dit->second + 1);
                if (inserted_src) {
                    queue.push_back(pre);
                    max_distance = std::max(max_distance, sit->second);
                }
            }
        }
    }
    return max_distance;
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 12;
    if (maxW < 4 || maxW > 16) return 2;

    std::vector<std::vector<Word>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 4; W <= maxW; ++W) {
        Rank depth1 = 0, depth4 = 0, depth7 = 0;
        int worst = 0;
        for (int i = 0; i <= W - 4; ++i) {
            Rank d1 = 0, d4 = 0, d7 = 0;
            for (const Word& u : words[W - 2]) {
                const Key seed = project_key(Key{'C', u}, i, W);
                const int got = local_component_depth(seed, W, i);
                const int expected = expected_seed_depth(u, i);
                if (got != expected)
                    fail("component depth classification W=" + std::to_string(W) +
                         " i=" + std::to_string(i) + " u=" + u +
                         " got=" + std::to_string(got) +
                         " expected=" + std::to_string(expected));
                worst = std::max(worst, got);
                if (got == 1) ++d1;
                else if (got == 4) ++d4;
                else if (got == 7) ++d7;
                else fail("unexpected component depth");
            }

            const Rank m2 = words[W - 2].size();
            const Rank m3 = words[W - 3].size();
            if (d1 != m3 || d4 != m2 - 2 * m3 || d7 != m3)
                fail("component depth count formula W=" + std::to_string(W));
            if (i == 0) {
                depth1 = d1;
                depth4 = d4;
                depth7 = d7;
            } else if (d1 != depth1 || d4 != depth4 || d7 != depth7) {
                fail("component depth position dependence W=" + std::to_string(W));
            }
        }

        std::cout << "W=" << W
                  << " depth1=" << depth1
                  << " depth4=" << depth4
                  << " depth7=" << depth7
                  << " worst_bipartite_distance=" << worst
                  << " inverse_expansion_rounds_max=3"
                  << " local_pair_classifier=1"
                  << " OK\n";
    }

    std::cout << "ALL_OK component_depth_bound=7"
              << " fixed_round_local_reconstruction=1\n";
    return 0;
}
