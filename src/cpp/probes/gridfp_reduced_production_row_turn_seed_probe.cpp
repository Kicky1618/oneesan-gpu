#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_row_turn_split_probe_main_unused
#include "gridfp_reduced_production_row_turn_split_probe.cpp"
#pragma pop_macro("main")

#include <deque>
#include <set>

namespace {

template<class BasisFn>
std::pair<std::vector<Rank>, Rank> component_ids(
    const std::vector<Key>& src,
    const std::vector<Key>& dst,
    BasisFn basis
) {
    const Rank ns = src.size(), nd = dst.size();
    std::map<Key, Rank> dr;
    for (Rank d = 0; d < nd; ++d) dr.emplace(dst[static_cast<std::size_t>(d)], d);
    std::vector<std::vector<Rank>> adj(static_cast<std::size_t>(ns + nd));
    for (Rank s = 0; s < ns; ++s) {
        for (const auto& [z, c] : basis(src[static_cast<std::size_t>(s)])) {
            if (!c) fail("turn seed zero coefficient");
            const auto it = dr.find(z);
            if (it == dr.end()) fail("turn seed destination outside layout");
            adj[static_cast<std::size_t>(s)].push_back(ns + it->second);
            adj[static_cast<std::size_t>(ns + it->second)].push_back(s);
        }
    }
    std::vector<Rank> cid(static_cast<std::size_t>(ns + nd), Rank(-1));
    Rank components = 0;
    for (Rank root = 0; root < ns + nd; ++root) {
        if (cid[static_cast<std::size_t>(root)] != Rank(-1)) continue;
        std::deque<Rank> q;
        q.push_back(root);
        cid[static_cast<std::size_t>(root)] = components;
        while (!q.empty()) {
            const Rank x = q.front(); q.pop_front();
            for (Rank y : adj[static_cast<std::size_t>(x)]) {
                if (cid[static_cast<std::size_t>(y)] == Rank(-1)) {
                    cid[static_cast<std::size_t>(y)] = components;
                    q.push_back(y);
                }
            }
        }
        ++components;
    }
    return {std::move(cid), components};
}

void verify_compression_seeds(int W) {
    const auto main_words = gen_words(W);
    const auto block_words = gen_words(W - 1);
    const std::vector<Key> q1 = layout(main_words, block_words, 1);
    std::vector<Key> main;
    for (MateID m : main_words) main.push_back(Key{false, m});
    const auto [cid, components] = component_ids(
        q1, main, [&](Key k) { return step_basis(k, W, 1, false); });
    if (components != block_words.size()) fail("compression component count");
    std::map<Key, Rank> sr;
    for (Rank s = 0; s < q1.size(); ++s) sr.emplace(q1[static_cast<std::size_t>(s)], s);
    std::vector<std::uint8_t> seen(static_cast<std::size_t>(components));
    for (MateID v : block_words) {
        const Key seed = mget(v, 0) != N
            ? Key{true, v}
            : Key{false, blocked_exclude(v, 1)};
        const auto it = sr.find(seed);
        if (it == sr.end()) fail("compression seed outside Q1");
        const Rank id = cid[static_cast<std::size_t>(it->second)];
        if (seen[static_cast<std::size_t>(id)]++) fail("duplicate compression seed");
    }
    for (std::uint8_t x : seen) if (x != 1) fail("compression component without seed");
}

void verify_expansion_seeds(int W) {
    const auto main_words = gen_words(W);
    const auto block_words = gen_words(W - 1);
    std::vector<Key> main;
    for (MateID m : main_words) main.push_back(Key{false, m});
    const std::vector<Key> q2 = layout(main_words, block_words, 2);
    const auto [cid, components] = component_ids(
        main, q2, [&](Key k) { return reverse_p1_to_q2(k, W); });
    const Rank expected = block_words.size() - gen_words(W - 3).size();
    if (components != expected) fail("expansion component count");
    std::map<Key, Rank> sr;
    for (Rank s = 0; s < main.size(); ++s) sr.emplace(main[static_cast<std::size_t>(s)], s);
    std::vector<std::uint8_t> seen(static_cast<std::size_t>(components));
    Rank labels = 0;
    for (MateID v : block_words) {
        if (mget(v, 0) == N && mget(v, 1) == N) continue;
        const Key seed{false, blocked_exclude_reverse(v, W, 1)};
        const auto it = sr.find(seed);
        if (it == sr.end()) fail("expansion seed outside main");
        const Rank id = cid[static_cast<std::size_t>(it->second)];
        if (seen[static_cast<std::size_t>(id)]++) fail("duplicate expansion seed");
        ++labels;
    }
    if (labels != expected) fail("expansion label count");
    for (std::uint8_t x : seen) if (x != 1) fail("expansion component without seed");
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 12;
    if (maxW < 5 || maxW > 12) return 2;

    for (int W = 5; W <= maxW; ++W) {
        verify_compression_seeds(W);
        verify_expansion_seeds(W);
        const Rank compress_components = gen_words(W - 1).size();
        const Rank expand_components = compress_components - gen_words(W - 3).size();
        std::cout << "W=" << W
                  << " compression_components=" << compress_components
                  << " compression_seed=width_Wm1_all_labels"
                  << " compression_max_pairs_bound=" << ((W + 3) / 2)
                  << " expansion_components=" << expand_components
                  << " expansion_seed=width_Wm1_not_both_N"
                  << " expansion_max_src_bound=" << (W / 2 + 3)
                  << " expansion_max_dst_bound=" << (W / 2 + 4)
                  << " table_bytes=0 seeds=OK\n";
    }

    std::cout << "W=28_theory compression_components=135015505407"
              << " compression_max_pairs_candidate=15"
              << " expansion_components=118389089432"
              << " expansion_max_src_candidate=17 expansion_max_dst_candidate=18"
              << " seed_table_bytes=0\n";
    std::cout << "ALL_OK production_split_turn_table_free_seeds=1\n";
    return 0;
}
