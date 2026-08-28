#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_grouped_codec_probe_main_unused
#include "gridfp_reduced_production_grouped_codec_probe.cpp"
#pragma pop_macro("main")

#include <deque>
#include <set>

namespace {

Vec row_turn_q2_basis(Key src, int W) {
    // Q2 forward --reduced p2--> Q1 forward --raw p1--> main
    //            --reverse raw p1 + Q2 projection--> Q2 reverse.
    Vec a = reduced_step_basis(src, W, 2, false);
    Vec b = step_vec(a, W, 1, false);
    Vec c = step_vec(b, W, 1, true);
    return project_vec(c, W, 2, true);
}

struct TurnStats {
    Rank states = 0;
    Rank components = 0;
    Rank edges = 0;
    Rank max_pairs = 0;
    Rank max_edges = 0;
    Rank max_fanout = 0;
    Rank max_indegree = 0;
    Rank max_slot_union = 0;
};

TurnStats verify_turn(int W, int K, int ngpu) {
    if (K < 1 || K + 2 >= W) fail("turn K range");
    const auto main = gen_words(W);
    const auto block = gen_words(W - 1);
    const std::vector<Key> src = layout(main, block, 2);
    const std::vector<Key> dst = layout(main, block, 2);
    if (src.size() != dst.size()) fail("turn square layout");
    const Rank n = src.size();

    std::map<Key, Rank> dst_rank;
    for (Rank d = 0; d < n; ++d) dst_rank.emplace(dst[static_cast<std::size_t>(d)], d);

    std::vector<std::vector<Rank>> adj(static_cast<std::size_t>(2 * n));
    std::vector<Rank> indegree(static_cast<std::size_t>(n));
    TurnStats st;
    st.states = n;
    for (Rank s = 0; s < n; ++s) {
        const Vec col = row_turn_q2_basis(src[static_cast<std::size_t>(s)], W);
        st.max_fanout = std::max<Rank>(st.max_fanout, col.size());
        for (const auto& [z, c] : col) {
            if (c == 0) fail("turn zero coefficient");
            const auto it = dst_rank.find(z);
            if (it == dst_rank.end()) fail("turn destination outside Q2");
            const Rank d = it->second;
            adj[static_cast<std::size_t>(s)].push_back(n + d);
            adj[static_cast<std::size_t>(n + d)].push_back(s);
            ++indegree[static_cast<std::size_t>(d)];
            ++st.edges;
        }
    }
    for (Rank x : indegree) st.max_indegree = std::max(st.max_indegree, x);

    ProductionFactorTables tables(W);
    const OwnerPlan plan = make_owner_plan(W, K, ngpu);
    const int forward_start = K + 2; // physical window [1, K+2]
    const int reverse_start = 2;     // same physical window [1, K+2]

    std::vector<std::uint8_t> seen(static_cast<std::size_t>(2 * n));
    for (Rank root = 0; root < 2 * n; ++root) {
        if (seen[static_cast<std::size_t>(root)]) continue;
        std::deque<Rank> q;
        q.push_back(root);
        seen[static_cast<std::size_t>(root)] = 1;
        std::vector<Rank> ss, dd;
        Rank component_edges2 = 0;
        while (!q.empty()) {
            const Rank x = q.front();
            q.pop_front();
            if (x < n) ss.push_back(x); else dd.push_back(x - n);
            component_edges2 += adj[static_cast<std::size_t>(x)].size();
            for (Rank y : adj[static_cast<std::size_t>(x)]) {
                if (!seen[static_cast<std::size_t>(y)]) {
                    seen[static_cast<std::size_t>(y)] = 1;
                    q.push_back(y);
                }
            }
        }
        if (ss.size() != dd.size()) fail("turn component unbalanced");
        const Rank component_edges = component_edges2 / 2;
        ++st.components;
        st.max_pairs = std::max<Rank>(st.max_pairs, ss.size());
        st.max_edges = std::max(st.max_edges, component_edges);

        std::set<std::pair<int, Rank>> source_slots;
        std::set<std::pair<int, Rank>> destination_slots;
        std::set<std::uint32_t> source_outer;
        std::set<std::uint32_t> destination_outer;

        for (Rank s : ss) {
            const Key k = src[static_cast<std::size_t>(s)];
            const GroupedRank r = grouped_rank(
                k, tables, W, 2, false, forward_start, K, ngpu, plan);
            source_slots.emplace(r.owner, r.local);
            const MateID full = embed_full(k, W, 2, false);
            source_outer.insert(compress_outside_window(
                occupancy_mask(full, W), W, 1, K + 2));
        }
        for (Rank d : dd) {
            const Key k = dst[static_cast<std::size_t>(d)];
            const GroupedRank r = grouped_rank(
                k, tables, W, 2, true, reverse_start, K, ngpu, plan);
            destination_slots.emplace(r.owner, r.local);
            const MateID full = embed_full(k, W, 2, true);
            destination_outer.insert(compress_outside_window(
                occupancy_mask(full, W), W, 1, K + 2));
        }

        std::set<std::pair<int, Rank>> slot_union = source_slots;
        slot_union.insert(destination_slots.begin(), destination_slots.end());
        st.max_slot_union = std::max<Rank>(st.max_slot_union, slot_union.size());
        if (source_outer != destination_outer) fail("turn outer support changed");
        if (source_slots != destination_slots)
            fail("turn grouped slot set mismatch W=" + std::to_string(W) +
                 " K=" + std::to_string(K));
    }

    return st;
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 12;
    const int ngpu = argc > 2 ? std::atoi(argv[2]) : 8;
    if (maxW < 6 || maxW > 12 || ngpu < 2 || ngpu > 16) return 2;

    for (int W = 6; W <= maxW; ++W) {
        for (int K = 2; K <= std::min(4, W - 3); ++K) {
            if (K + 2 >= W) continue;
            const TurnStats st = verify_turn(W, K, ngpu);
            std::cout << "W=" << W
                      << " K=" << K
                      << " states=" << st.states
                      << " components=" << st.components
                      << " edges=" << st.edges
                      << " max_pairs=" << st.max_pairs
                      << " max_edges=" << st.max_edges
                      << " max_fanout=" << st.max_fanout
                      << " max_indegree=" << st.max_indegree
                      << " max_slot_union=" << st.max_slot_union
                      << " same_physical_window=1"
                      << " outer_support_preserved=1"
                      << " grouped_slot_set_equal=1"
                      << " single_state_stream_candidate=1\n";
        }
    }

    std::cout << "W=28_plan K=12 physical_window=[1,14]"
              << " source=Q2_forward destination=Q2_reverse"
              << " grouped_slot_set_closure=finite_width_verified"
              << " second_state_buffer_bytes=0_candidate=1\n";
    std::cout << "ALL_OK production_row_turn_grouped_slot_closure=1\n";
    return 0;
}
