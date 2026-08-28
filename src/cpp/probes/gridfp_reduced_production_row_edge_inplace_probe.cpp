#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_inverse_probe_main_unused
#include "gridfp_reduced_production_inverse_probe.cpp"
#pragma pop_macro("main")

#include <deque>
#include <limits>
#include <set>

namespace {

struct LocalEdge {
    int source = -1;
    int dest = -1;
    Coef coef = 0;
};

struct PenultimateStats {
    Rank components = 0;
    Rank states = 0;
    Rank max_pairs = 0;
    Rank negative_matching = 0;
    Rank correction_edges = 0;
    Rank max_correction_out = 0;
    Rank max_depth = 0;
};

PenultimateStats verify_penultimate_inplace(
    const std::vector<MateID>& main,
    const std::vector<MateID>& block,
    int W,
    bool reverse
) {
    const int p = reverse ? W - 2 : 2;
    const int next = reverse ? W - 1 : 1;
    const auto src = layout(main, block, p);
    const auto dst = layout(main, block, next);
    if (src.size() != dst.size()) fail("row-edge penultimate non-square layout");

    std::map<Key, int> sidx, didx;
    for (int i = 0; i < static_cast<int>(src.size()); ++i) sidx.emplace(src[static_cast<std::size_t>(i)], i);
    for (int i = 0; i < static_cast<int>(dst.size()); ++i) didx.emplace(dst[static_cast<std::size_t>(i)], i);

    std::vector<Vec> columns(src.size());
    std::vector<Vec> incoming(dst.size());
    for (int s = 0; s < static_cast<int>(src.size()); ++s) {
        columns[static_cast<std::size_t>(s)] = reduced_step_basis(src[static_cast<std::size_t>(s)], W, p, reverse);
        for (const auto& [d, c] : columns[static_cast<std::size_t>(s)]) {
            if (c != 1 && c != -1) fail("row-edge penultimate coefficient");
            const auto it = didx.find(d);
            if (it == didx.end()) fail("row-edge penultimate destination outside layout");
            add(incoming[static_cast<std::size_t>(it->second)], src[static_cast<std::size_t>(s)], c);
        }
    }

    std::vector<std::uint8_t> seen_s(src.size());
    std::vector<std::uint8_t> seen_d(dst.size());
    PenultimateStats st;
    st.states = src.size();

    for (int seed = 0; seed < static_cast<int>(src.size()); ++seed) {
        if (seen_s[static_cast<std::size_t>(seed)]) continue;
        std::vector<int> ss, dd;
        std::deque<std::pair<bool, int>> q;
        q.push_back({false, seed});
        seen_s[static_cast<std::size_t>(seed)] = 1;
        while (!q.empty()) {
            const auto [right, v] = q.front();
            q.pop_front();
            if (!right) {
                ss.push_back(v);
                for (const auto& [d, c] : columns[static_cast<std::size_t>(v)]) {
                    (void)c;
                    const int z = didx.at(d);
                    if (!seen_d[static_cast<std::size_t>(z)]) {
                        seen_d[static_cast<std::size_t>(z)] = 1;
                        q.push_back({true, z});
                    }
                }
            } else {
                dd.push_back(v);
                for (const auto& [s, c] : incoming[static_cast<std::size_t>(v)]) {
                    (void)c;
                    const int z = sidx.at(s);
                    if (!seen_s[static_cast<std::size_t>(z)]) {
                        seen_s[static_cast<std::size_t>(z)] = 1;
                        q.push_back({false, z});
                    }
                }
            }
        }
        if (ss.size() != dd.size()) fail("row-edge penultimate unbalanced component");
        const int k = static_cast<int>(ss.size());
        ++st.components;
        st.max_pairs = std::max<Rank>(st.max_pairs, k);

        std::map<int, int> slocal, dlocal;
        for (int i = 0; i < k; ++i) {
            slocal.emplace(ss[static_cast<std::size_t>(i)], i);
            dlocal.emplace(dd[static_cast<std::size_t>(i)], i);
        }
        std::vector<LocalEdge> edges;
        std::vector<std::vector<int>> sout(static_cast<std::size_t>(k));
        std::vector<std::vector<int>> din(static_cast<std::size_t>(k));
        for (int ls = 0; ls < k; ++ls) {
            const int sg = ss[static_cast<std::size_t>(ls)];
            for (const auto& [d, c] : columns[static_cast<std::size_t>(sg)]) {
                const int dg = didx.at(d);
                const int ld = dlocal.at(dg);
                const int ei = static_cast<int>(edges.size());
                edges.push_back(LocalEdge{ls, ld, c});
                sout[static_cast<std::size_t>(ls)].push_back(ei);
                din[static_cast<std::size_t>(ld)].push_back(ei);
            }
        }

        std::vector<int> sdeg(static_cast<std::size_t>(k)), ddeg(static_cast<std::size_t>(k));
        std::vector<std::uint8_t> live_s(static_cast<std::size_t>(k), 1), live_d(static_cast<std::size_t>(k), 1);
        std::deque<std::pair<bool, int>> leaves;
        for (int i = 0; i < k; ++i) {
            sdeg[static_cast<std::size_t>(i)] = static_cast<int>(sout[static_cast<std::size_t>(i)].size());
            ddeg[static_cast<std::size_t>(i)] = static_cast<int>(din[static_cast<std::size_t>(i)].size());
            if (sdeg[static_cast<std::size_t>(i)] == 1) leaves.push_back({false, i});
            if (ddeg[static_cast<std::size_t>(i)] == 1) leaves.push_back({true, i});
        }

        std::vector<int> match_d(static_cast<std::size_t>(k), -1), match_s(static_cast<std::size_t>(k), -1);
        std::vector<Coef> match_coef(static_cast<std::size_t>(k));
        int matched = 0;
        while (!leaves.empty()) {
            const auto [right, v] = leaves.front();
            leaves.pop_front();
            int s = -1, d = -1, ei = -1;
            if (!right) {
                if (!live_s[static_cast<std::size_t>(v)] || sdeg[static_cast<std::size_t>(v)] != 1) continue;
                s = v;
                for (int z : sout[static_cast<std::size_t>(s)]) {
                    const int x = edges[static_cast<std::size_t>(z)].dest;
                    if (live_d[static_cast<std::size_t>(x)]) { d = x; ei = z; break; }
                }
            } else {
                if (!live_d[static_cast<std::size_t>(v)] || ddeg[static_cast<std::size_t>(v)] != 1) continue;
                d = v;
                for (int z : din[static_cast<std::size_t>(d)]) {
                    const int x = edges[static_cast<std::size_t>(z)].source;
                    if (live_s[static_cast<std::size_t>(x)]) { s = x; ei = z; break; }
                }
            }
            if (s < 0 || d < 0 || ei < 0) continue;
            if (match_d[static_cast<std::size_t>(s)] >= 0 || match_s[static_cast<std::size_t>(d)] >= 0)
                fail("row-edge penultimate duplicate matching");
            match_d[static_cast<std::size_t>(s)] = d;
            match_s[static_cast<std::size_t>(d)] = s;
            match_coef[static_cast<std::size_t>(s)] = edges[static_cast<std::size_t>(ei)].coef;
            st.negative_matching += match_coef[static_cast<std::size_t>(s)] < 0;
            ++matched;
            live_s[static_cast<std::size_t>(s)] = 0;
            live_d[static_cast<std::size_t>(d)] = 0;
            for (int z : sout[static_cast<std::size_t>(s)]) {
                const int x = edges[static_cast<std::size_t>(z)].dest;
                if (live_d[static_cast<std::size_t>(x)] && --ddeg[static_cast<std::size_t>(x)] == 1)
                    leaves.push_back({true, x});
            }
            for (int z : din[static_cast<std::size_t>(d)]) {
                const int x = edges[static_cast<std::size_t>(z)].source;
                if (live_s[static_cast<std::size_t>(x)] && --sdeg[static_cast<std::size_t>(x)] == 1)
                    leaves.push_back({false, x});
            }
        }
        if (matched != k) fail("row-edge penultimate matching not forced");

        std::vector<std::vector<int>> precedence(static_cast<std::size_t>(k));
        std::vector<int> indeg(static_cast<std::size_t>(k));
        std::vector<int> corr_out(static_cast<std::size_t>(k));
        for (const auto& e : edges) {
            const int target = match_s[static_cast<std::size_t>(e.dest)];
            if (target < 0) fail("row-edge penultimate unmatched destination");
            if (target == e.source) continue;
            precedence[static_cast<std::size_t>(target)].push_back(e.source);
            ++indeg[static_cast<std::size_t>(e.source)];
            ++corr_out[static_cast<std::size_t>(e.source)];
            ++st.correction_edges;
        }
        for (int x : corr_out) st.max_correction_out = std::max<Rank>(st.max_correction_out, x);

        std::deque<int> ready;
        std::vector<int> depth(static_cast<std::size_t>(k));
        for (int i = 0; i < k; ++i) if (!indeg[static_cast<std::size_t>(i)]) ready.push_back(i);
        int visited = 0;
        while (!ready.empty()) {
            const int v = ready.front();
            ready.pop_front();
            ++visited;
            st.max_depth = std::max<Rank>(st.max_depth, depth[static_cast<std::size_t>(v)]);
            for (int z : precedence[static_cast<std::size_t>(v)]) {
                depth[static_cast<std::size_t>(z)] = std::max(depth[static_cast<std::size_t>(z)], depth[static_cast<std::size_t>(v)] + 1);
                if (--indeg[static_cast<std::size_t>(z)] == 0) ready.push_back(z);
            }
        }
        if (visited != k) fail("row-edge penultimate correction cycle");
    }

    for (std::uint8_t x : seen_s) if (!x) fail("row-edge penultimate source coverage");
    for (std::uint8_t x : seen_d) if (!x) fail("row-edge penultimate destination coverage");
    if (st.max_correction_out > 2 || st.max_depth > 4)
        fail("row-edge penultimate fixed-wave bound");
    if (st.max_pairs > Rank(W / 2 + 4)) fail("row-edge penultimate pair bound");
    return st;
}

struct FinalStats {
    Rank components = 0;
    Rank lossy_components = 0;
    Rank square_components = 0;
    Rank max_sources = 0;
    Rank max_destinations = 0;
    Rank max_indegree = 0;
};

FinalStats verify_final_contraction(
    const std::vector<MateID>& main,
    const std::vector<MateID>& block,
    int W,
    bool reverse
) {
    const int p = reverse ? W - 1 : 1;
    const auto src = layout(main, block, p);
    std::map<Key, int> sidx;
    std::map<MateID, int> didx;
    for (int i = 0; i < static_cast<int>(src.size()); ++i) sidx.emplace(src[static_cast<std::size_t>(i)], i);
    for (int i = 0; i < static_cast<int>(main.size()); ++i) didx.emplace(main[static_cast<std::size_t>(i)], i);

    std::vector<std::vector<int>> sout(src.size());
    std::vector<std::vector<int>> din(main.size());
    for (int s = 0; s < static_cast<int>(src.size()); ++s) {
        const Vec col = step_basis(src[static_cast<std::size_t>(s)], W, p, reverse);
        for (const auto& [d, c] : col) {
            if (c != 1 || d.blocked) fail("row-edge final non-main/nonunit edge");
            const auto it = didx.find(d.mate);
            if (it == didx.end()) fail("row-edge final destination outside main layout");
            sout[static_cast<std::size_t>(s)].push_back(it->second);
            din[static_cast<std::size_t>(it->second)].push_back(s);
        }
    }

    std::vector<std::uint8_t> seen_s(src.size()), seen_d(main.size());
    std::vector<Coef> input(src.size()), work(src.size()), reference(main.size());
    for (int s = 0; s < static_cast<int>(src.size()); ++s) {
        input[static_cast<std::size_t>(s)] = Coef(1 + ((s * 131 + W * 17 + int(reverse) * 19) % 1009));
        work[static_cast<std::size_t>(s)] = input[static_cast<std::size_t>(s)];
        for (int d : sout[static_cast<std::size_t>(s)]) reference[static_cast<std::size_t>(d)] += input[static_cast<std::size_t>(s)];
    }

    FinalStats st;
    for (int seed = 0; seed < static_cast<int>(src.size()); ++seed) {
        if (seen_s[static_cast<std::size_t>(seed)]) continue;
        std::vector<int> ss, dd;
        std::deque<std::pair<bool, int>> q;
        q.push_back({false, seed});
        seen_s[static_cast<std::size_t>(seed)] = 1;
        while (!q.empty()) {
            const auto [right, v] = q.front();
            q.pop_front();
            if (!right) {
                ss.push_back(v);
                for (int d : sout[static_cast<std::size_t>(v)]) {
                    if (!seen_d[static_cast<std::size_t>(d)]) {
                        seen_d[static_cast<std::size_t>(d)] = 1;
                        q.push_back({true, d});
                    }
                }
            } else {
                dd.push_back(v);
                for (int s : din[static_cast<std::size_t>(v)]) {
                    if (!seen_s[static_cast<std::size_t>(s)]) {
                        seen_s[static_cast<std::size_t>(s)] = 1;
                        q.push_back({false, s});
                    }
                }
            }
        }

        ++st.components;
        st.max_sources = std::max<Rank>(st.max_sources, ss.size());
        st.max_destinations = std::max<Rank>(st.max_destinations, dd.size());
        int blocked_sources = 0;
        std::set<MateID> main_sources, destinations;
        for (int s : ss) {
            const Key k = src[static_cast<std::size_t>(s)];
            if (k.blocked) ++blocked_sources;
            else main_sources.insert(k.mate);
        }
        for (int d : dd) {
            destinations.insert(main[static_cast<std::size_t>(d)]);
            st.max_indegree = std::max<Rank>(st.max_indegree, din[static_cast<std::size_t>(d)].size());
        }
        if (main_sources != destinations)
            fail("row-edge final main source/destination set mismatch");
        if (blocked_sources == 0) {
            if (ss.size() != dd.size()) fail("row-edge final square component shape");
            ++st.square_components;
        } else if (blocked_sources == 1) {
            if (ss.size() != 3 || dd.size() != 2) fail("row-edge final lossy component not 3-to-2");
            ++st.lossy_components;
        } else {
            fail("row-edge final multiple blocked sources in component");
        }

        // Component-local in-place schedule: all source values are loaded before
        // any main slot in this component is overwritten. Main source and main
        // destination keys are exactly the same set, so no other component can
        // need a slot written here.
        std::map<int, Coef> local_out;
        for (int s : ss)
            for (int d : sout[static_cast<std::size_t>(s)])
                local_out[d] += input[static_cast<std::size_t>(s)];
        for (const auto& [d, value] : local_out) {
            const MateID mate = main[static_cast<std::size_t>(d)];
            const auto it = sidx.find(Key{false, mate});
            if (it == sidx.end()) fail("row-edge final main rank missing from Q1");
            work[static_cast<std::size_t>(it->second)] = value;
        }
    }

    for (std::uint8_t x : seen_s) if (!x) fail("row-edge final source coverage");
    for (std::uint8_t x : seen_d) if (!x) fail("row-edge final destination coverage");
    for (int d = 0; d < static_cast<int>(main.size()); ++d) {
        const int sr = sidx.at(Key{false, main[static_cast<std::size_t>(d)]});
        if (work[static_cast<std::size_t>(sr)] != reference[static_cast<std::size_t>(d)])
            fail("row-edge final in-place arithmetic mismatch");
    }

    const Rank want_components = block.size();
    Rank canonical_blocked = 0;
    for (MateID b : block) if (mget(b, p - 1) != N) ++canonical_blocked;
    if (st.components != want_components || st.lossy_components != canonical_blocked)
        fail("row-edge final component count formula");
    if (st.max_sources > Rank((W + 1) / 2 + 1)) fail("row-edge final source bound");
    return st;
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 11;
    if (maxW < 5 || maxW > 12) return 2;

    std::vector<std::vector<MateID>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 5; W <= maxW; ++W) {
        for (bool reverse : {false, true}) {
            const PenultimateStats a = verify_penultimate_inplace(words[W], words[W - 1], W, reverse);
            const FinalStats b = verify_final_contraction(words[W], words[W - 1], W, reverse);
            std::cout << "W=" << W
                      << " direction=" << (reverse ? "reverse" : "forward")
                      << " q2_q1_states=" << a.states
                      << " q2_q1_components=" << a.components
                      << " q2_q1_max_pairs=" << a.max_pairs
                      << " q2_q1_max_depth=" << a.max_depth
                      << " q2_q1_max_correction_out=" << a.max_correction_out
                      << " final_components=" << b.components
                      << " final_lossy=" << b.lossy_components
                      << " final_square=" << b.square_components
                      << " final_max_sources=" << b.max_sources
                      << " final_max_destinations=" << b.max_destinations
                      << " final_max_indegree=" << b.max_indegree
                      << " main_slots_same_component=1"
                      << " full_stream_scratch=0"
                      << " in_place_candidate=1 OK\n";
        }
    }

    std::cout << "W=28_theory q2_q1_max_pairs_bound=18"
              << " q2_q1_fixed_waves=5"
              << " final_components=135015505407"
              << " final_lossy_components=87677551081"
              << " final_max_source_slots=15"
              << " second_full_stream=0"
              << "\n";
    std::cout << "ALL_OK reduced_row_edge_inplace_decomposition=1\n";
    return 0;
}
