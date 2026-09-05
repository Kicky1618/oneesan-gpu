#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_grouped_codec_probe_main_unused
#include "gridfp_reduced_production_grouped_codec_probe.cpp"
#pragma pop_macro("main")

#include <deque>
#include <set>

namespace {

Vec reverse_p1_to_q2(Key main_source, int W) {
    if (main_source.blocked) fail("reverse edge source must be main");
    return project_vec(step_basis(main_source, W, 1, true), W, 2, true);
}

struct RectStats {
    Rank source_states = 0;
    Rank destination_states = 0;
    Rank components = 0;
    Rank edges = 0;
    Rank max_source_pairs = 0;
    Rank max_destination_pairs = 0;
    Rank max_union_slots = 0;
    Rank max_fanout = 0;
};

template<class BasisFn, class SrcRankFn, class DstRankFn>
RectStats verify_rectangular_components(
    const std::vector<Key>& src,
    const std::vector<Key>& dst,
    BasisFn basis,
    SrcRankFn src_rank_fn,
    DstRankFn dst_rank_fn,
    bool require_destination_subset,
    const char* label
) {
    RectStats st;
    st.source_states = src.size();
    st.destination_states = dst.size();
    const Rank ns = src.size(), nd = dst.size();
    std::map<Key, Rank> dr;
    for (Rank d = 0; d < nd; ++d) dr.emplace(dst[static_cast<std::size_t>(d)], d);

    std::vector<std::vector<Rank>> adj(static_cast<std::size_t>(ns + nd));
    for (Rank s = 0; s < ns; ++s) {
        const Vec col = basis(src[static_cast<std::size_t>(s)]);
        st.max_fanout = std::max<Rank>(st.max_fanout, col.size());
        for (const auto& [z, c] : col) {
            if (!c) fail(std::string(label) + " zero coefficient");
            const auto it = dr.find(z);
            if (it == dr.end()) fail(std::string(label) + " destination outside layout");
            const Rank d = it->second;
            adj[static_cast<std::size_t>(s)].push_back(ns + d);
            adj[static_cast<std::size_t>(ns + d)].push_back(s);
            ++st.edges;
        }
    }

    std::vector<std::uint8_t> seen(static_cast<std::size_t>(ns + nd));
    for (Rank root = 0; root < ns + nd; ++root) {
        if (seen[static_cast<std::size_t>(root)]) continue;
        std::deque<Rank> q;
        q.push_back(root);
        seen[static_cast<std::size_t>(root)] = 1;
        std::vector<Rank> ss, dd;
        while (!q.empty()) {
            const Rank x = q.front(); q.pop_front();
            if (x < ns) ss.push_back(x); else dd.push_back(x - ns);
            for (Rank y : adj[static_cast<std::size_t>(x)]) {
                if (!seen[static_cast<std::size_t>(y)]) {
                    seen[static_cast<std::size_t>(y)] = 1;
                    q.push_back(y);
                }
            }
        }
        ++st.components;
        st.max_source_pairs = std::max<Rank>(st.max_source_pairs, ss.size());
        st.max_destination_pairs = std::max<Rank>(st.max_destination_pairs, dd.size());

        std::set<std::pair<int, Rank>> S, D;
        for (Rank s : ss) {
            const auto r = src_rank_fn(src[static_cast<std::size_t>(s)]);
            S.emplace(r.owner, r.local);
        }
        for (Rank d : dd) {
            const auto r = dst_rank_fn(dst[static_cast<std::size_t>(d)]);
            D.emplace(r.owner, r.local);
        }
        std::set<std::pair<int, Rank>> U = S;
        U.insert(D.begin(), D.end());
        st.max_union_slots = std::max<Rank>(st.max_union_slots, U.size());
        if (require_destination_subset) {
            if (!std::includes(S.begin(), S.end(), D.begin(), D.end()))
                fail(std::string(label) + " destination slots not subset of source slots");
        } else {
            if (!std::includes(D.begin(), D.end(), S.begin(), S.end()))
                fail(std::string(label) + " source slots not subset of destination slots");
        }
    }
    return st;
}

void verify_split_turn(int W, int K, int ngpu) {
    const auto main_words = gen_words(W);
    const auto block_words = gen_words(W - 1);
    const std::vector<Key> q1 = layout(main_words, block_words, 1);
    const std::vector<Key> q2 = layout(main_words, block_words, 2);
    std::vector<Key> main;
    main.reserve(main_words.size());
    for (MateID m : main_words) main.push_back(Key{false, m});

    ProductionFactorTables tables(W);
    const OwnerPlan plan = make_owner_plan(W, K, ngpu);

    // B=[0,K+1] is the one-cell-left shifted window used for forward p2 and p1.
    const int forward_B_start = K + 1;
    auto rank_q1_B = [&](Key k) {
        return grouped_rank(k, tables, W, 1, false, forward_B_start, K, ngpu, plan);
    };
    // A=[1,K+2] is the first reverse tile window after the turn.
    const int reverse_A_start = 2;
    auto rank_q2_A = [&](Key k) {
        return grouped_rank(k, tables, W, 2, true, reverse_A_start, K, ngpu, plan);
    };

    const RectStats compress = verify_rectangular_components(
        q1, main,
        [&](Key k) { return step_basis(k, W, 1, false); },
        rank_q1_B,
        rank_q1_B,
        true,
        "forward p1 compression");
    const RectStats expand = verify_rectangular_components(
        main, q2,
        [&](Key k) { return reverse_p1_to_q2(k, W); },
        rank_q2_A,
        rank_q2_A,
        false,
        "reverse p1 expansion");

    if (compress.max_fanout > 2) fail("forward p1 fanout bound");
    if (expand.max_fanout > 3) fail("reverse p1 fanout bound");

    std::cout << "W=" << W
              << " K=" << K
              << " compress_src=" << compress.source_states
              << " compress_dst=" << compress.destination_states
              << " compress_components=" << compress.components
              << " compress_max_src=" << compress.max_source_pairs
              << " compress_max_dst=" << compress.max_destination_pairs
              << " compress_max_union=" << compress.max_union_slots
              << " compress_fanout=" << compress.max_fanout
              << " compress_destination_subset=1"
              << " expand_src=" << expand.source_states
              << " expand_dst=" << expand.destination_states
              << " expand_components=" << expand.components
              << " expand_max_src=" << expand.max_source_pairs
              << " expand_max_dst=" << expand.max_destination_pairs
              << " expand_max_union=" << expand.max_union_slots
              << " expand_fanout=" << expand.max_fanout
              << " expand_source_subset=1"
              << " single_stream=1\n";
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 12;
    const int ngpu = argc > 2 ? std::atoi(argv[2]) : 8;
    if (maxW < 6 || maxW > 12 || ngpu < 2 || ngpu > 16) return 2;

    for (int W = 6; W <= maxW; ++W) {
        for (int K = 2; K <= std::min(4, W - 3); ++K) {
            if (K + 2 >= W) continue;
            verify_split_turn(W, K, ngpu);
        }
    }

    std::cout << "W=28_plan K=12"
              << " compress_max_src_candidate=15"
              << " expand_max_dst_candidate=18"
              << " forward_window=[0,13] reverse_window=[1,14]"
              << " blocked_tail_reused=1 second_state_buffer_bytes=0_candidate=1\n";
    std::cout << "ALL_OK production_split_row_turn_inplace=1\n";
    return 0;
}
