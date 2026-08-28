#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_inverse_probe_main_unused
#include "gridfp_reduced_production_inverse_probe.cpp"
#pragma pop_macro("main")

#include <deque>
#include <limits>
#include <set>

namespace {

struct PEdge { int s = -1, d = -1; Coef c = 0; };

std::vector<Key> production_component_seeds(
    const std::vector<MateID>& labels, int W, int p, bool reverse
) {
    std::vector<Key> out;
    for (MateID v : labels) {
        const int other = reverse ? p : p - 2;
        if (mget(v, p - 1) == N && mget(v, other) == N) continue;
        if (mget(v, p - 1) != N) out.push_back(Key{true, v});
        else out.push_back(Key{false, reverse
            ? blocked_exclude_reverse(v, W, p)
            : blocked_exclude(v, p)});
    }
    return out;
}

std::vector<MateValue> primitive_word(Key k, int W) {
    const int len = k.blocked ? W - 1 : W;
    std::vector<MateValue> out;
    for (int pos = 0; pos < len; ++pos) {
        const MateValue v = mget(k.mate, len - 1 - pos);
        if (v != N) out.push_back(v);
    }
    return out;
}

bool erase_one_lr(
    const std::vector<MateValue>& longer,
    const std::vector<MateValue>& shorter
) {
    if (longer.size() != shorter.size() + 2) return false;
    for (std::size_t q = 0; q + 1 < longer.size(); ++q) {
        if (longer[q] != L || longer[q + 1] != R) continue;
        std::vector<MateValue> z;
        z.reserve(shorter.size());
        z.insert(z.end(), longer.begin(), longer.begin() + static_cast<std::ptrdiff_t>(q));
        z.insert(z.end(), longer.begin() + static_cast<std::ptrdiff_t>(q + 2), longer.end());
        if (z == shorter) return true;
    }
    return false;
}

struct NilpotentStats {
    Rank components = 0;
    Rank states = 0;
    Rank edges = 0;
    Rank correction_edges = 0;
    Rank primitive_same = 0;
    Rank primitive_delete_lr = 0;
    Rank primitive_insert_lr_neg = 0;
    Rank max_pairs = 0;
    Rank max_depth = 0;
    Rank max_in = 0;
    Rank max_out = 0;
};

NilpotentStats verify_position(
    const std::vector<MateID>& main,
    const std::vector<MateID>& block,
    int W,
    int p,
    bool reverse
) {
    const int next = reverse ? p + 1 : p - 1;
    const auto src = layout(main, block, p);
    const auto dst = layout(main, block, next);
    if (src.size() != dst.size()) fail("nilpotent non-square layout");
    const Rank n = src.size();
    std::map<Key, Rank> srank, drank;
    for (Rank r = 0; r < n; ++r) {
        srank.emplace(src[static_cast<std::size_t>(r)], r);
        drank.emplace(dst[static_cast<std::size_t>(r)], r);
    }

    NilpotentStats st;
    st.states = n;
    const auto seeds = production_component_seeds(block, W, p, reverse);
    st.components = seeds.size();
    std::vector<std::uint8_t> seen_s(static_cast<std::size_t>(n));
    std::vector<std::uint8_t> seen_d(static_cast<std::size_t>(n));

    for (Key seed : seeds) {
        std::set<Key> skeys, dkeys;
        std::deque<Key> q;
        skeys.insert(seed);
        q.push_back(seed);
        while (!q.empty()) {
            const Key s = q.front(); q.pop_front();
            for (const auto& [d, c] : reduced_step_basis(s, W, p, reverse)) {
                if (c != 1 && c != -1) fail("nilpotent edge coefficient");
                if (!dkeys.insert(d).second) continue;
                for (const auto& [pre, a] : inverse_reduced(d, W, p, reverse)) {
                    if (a != 1 && a != -1) fail("nilpotent inverse coefficient");
                    if (skeys.insert(pre).second) q.push_back(pre);
                }
            }
        }
        if (skeys.size() != dkeys.size()) fail("nilpotent component imbalance");
        const int k = static_cast<int>(skeys.size());
        st.max_pairs = std::max<Rank>(st.max_pairs, k);
        std::vector<Key> ss(skeys.begin(), skeys.end()), dd(dkeys.begin(), dkeys.end());
        std::map<Key, int> slocal, dlocal;
        for (int x = 0; x < k; ++x) {
            slocal.emplace(ss[static_cast<std::size_t>(x)], x);
            dlocal.emplace(dd[static_cast<std::size_t>(x)], x);
            const Rank a = srank.at(ss[static_cast<std::size_t>(x)]);
            const Rank b = drank.at(dd[static_cast<std::size_t>(x)]);
            if (seen_s[static_cast<std::size_t>(a)]++) fail("nilpotent source overlap");
            if (seen_d[static_cast<std::size_t>(b)]++) fail("nilpotent dest overlap");
        }

        std::vector<PEdge> edge;
        std::vector<std::vector<int>> sout(static_cast<std::size_t>(k));
        std::vector<std::vector<int>> din(static_cast<std::size_t>(k));
        for (int s = 0; s < k; ++s) {
            for (const auto& [dkey, c] : reduced_step_basis(ss[static_cast<std::size_t>(s)], W, p, reverse)) {
                const int d = dlocal.at(dkey);
                const int ei = static_cast<int>(edge.size());
                edge.push_back(PEdge{s, d, c});
                sout[static_cast<std::size_t>(s)].push_back(ei);
                din[static_cast<std::size_t>(d)].push_back(ei);
            }
        }
        st.edges += edge.size();

        std::vector<int> sdeg(static_cast<std::size_t>(k)), ddeg(static_cast<std::size_t>(k));
        std::vector<std::uint8_t> live_s(static_cast<std::size_t>(k), 1), live_d(static_cast<std::size_t>(k), 1);
        std::deque<std::pair<bool, int>> leaf;
        for (int x = 0; x < k; ++x) {
            sdeg[static_cast<std::size_t>(x)] = static_cast<int>(sout[static_cast<std::size_t>(x)].size());
            ddeg[static_cast<std::size_t>(x)] = static_cast<int>(din[static_cast<std::size_t>(x)].size());
            if (sdeg[static_cast<std::size_t>(x)] == 1) leaf.push_back({false, x});
            if (ddeg[static_cast<std::size_t>(x)] == 1) leaf.push_back({true, x});
        }
        std::vector<int> match_d(static_cast<std::size_t>(k), -1), match_s(static_cast<std::size_t>(k), -1);
        std::vector<Coef> match_c(static_cast<std::size_t>(k));
        int matched = 0;
        while (!leaf.empty()) {
            const auto [right, x] = leaf.front(); leaf.pop_front();
            int s = -1, d = -1, ei = -1;
            if (!right) {
                if (!live_s[static_cast<std::size_t>(x)] || sdeg[static_cast<std::size_t>(x)] != 1) continue;
                s = x;
                for (int z : sout[static_cast<std::size_t>(s)]) {
                    const int y = edge[static_cast<std::size_t>(z)].d;
                    if (live_d[static_cast<std::size_t>(y)]) { d = y; ei = z; break; }
                }
            } else {
                if (!live_d[static_cast<std::size_t>(x)] || ddeg[static_cast<std::size_t>(x)] != 1) continue;
                d = x;
                for (int z : din[static_cast<std::size_t>(d)]) {
                    const int y = edge[static_cast<std::size_t>(z)].s;
                    if (live_s[static_cast<std::size_t>(y)]) { s = y; ei = z; break; }
                }
            }
            if (s < 0 || d < 0 || ei < 0) continue;
            const Coef c = edge[static_cast<std::size_t>(ei)].c;
            if (c != 1 && c != -1) fail("nilpotent pivot coefficient");
            match_d[static_cast<std::size_t>(s)] = d;
            match_s[static_cast<std::size_t>(d)] = s;
            match_c[static_cast<std::size_t>(s)] = c;
            live_s[static_cast<std::size_t>(s)] = 0;
            live_d[static_cast<std::size_t>(d)] = 0;
            ++matched;
            for (int z : sout[static_cast<std::size_t>(s)]) {
                const int y = edge[static_cast<std::size_t>(z)].d;
                if (live_d[static_cast<std::size_t>(y)] && --ddeg[static_cast<std::size_t>(y)] == 1)
                    leaf.push_back({true, y});
            }
            for (int z : din[static_cast<std::size_t>(d)]) {
                const int y = edge[static_cast<std::size_t>(z)].s;
                if (live_s[static_cast<std::size_t>(y)] && --sdeg[static_cast<std::size_t>(y)] == 1)
                    leaf.push_back({false, y});
            }
        }
        if (matched != k) fail("nilpotent matching not fully forced");

        for (int s = 0; s < k; ++s) {
            const int d = match_d[static_cast<std::size_t>(s)];
            const auto ps = primitive_word(ss[static_cast<std::size_t>(s)], W);
            const auto pd = primitive_word(dd[static_cast<std::size_t>(d)], W);
            const Coef c = match_c[static_cast<std::size_t>(s)];
            if (ps == pd && c == 1) ++st.primitive_same;
            else if (erase_one_lr(ps, pd) && c == 1) ++st.primitive_delete_lr;
            else if (erase_one_lr(pd, ps) && c == -1) ++st.primitive_insert_lr_neg;
            else fail("nilpotent matching outside primitive three-class rule");
        }

        std::vector<std::vector<int>> corr(static_cast<std::size_t>(k));
        std::vector<int> indeg(static_cast<std::size_t>(k));
        for (const auto& e : edge) {
            const int target = match_s[static_cast<std::size_t>(e.d)];
            if (target == e.s) continue;
            corr[static_cast<std::size_t>(e.s)].push_back(target);
            ++indeg[static_cast<std::size_t>(target)];
            ++st.correction_edges;
        }
        for (int x = 0; x < k; ++x) {
            st.max_out = std::max<Rank>(st.max_out, corr[static_cast<std::size_t>(x)].size());
            st.max_in = std::max<Rank>(st.max_in, indeg[static_cast<std::size_t>(x)]);
        }

        std::deque<int> ready;
        std::vector<int> depth(static_cast<std::size_t>(k));
        for (int x = 0; x < k; ++x) if (!indeg[static_cast<std::size_t>(x)]) ready.push_back(x);
        int visited = 0;
        while (!ready.empty()) {
            const int u = ready.front(); ready.pop_front();
            ++visited;
            for (int v : corr[static_cast<std::size_t>(u)]) {
                depth[static_cast<std::size_t>(v)] = std::max(
                    depth[static_cast<std::size_t>(v)], depth[static_cast<std::size_t>(u)] + 1);
                if (--indeg[static_cast<std::size_t>(v)] == 0) ready.push_back(v);
            }
        }
        if (visited != k) fail("nilpotent correction cycle");
        for (int d : depth) st.max_depth = std::max<Rank>(st.max_depth, d);
        if (st.max_depth > 4) fail("nilpotent correction depth exceeded four");
    }

    for (Rank r = 0; r < n; ++r)
        if (seen_s[static_cast<std::size_t>(r)] != 1 || seen_d[static_cast<std::size_t>(r)] != 1)
            fail("nilpotent component coverage");
    if (st.states + st.correction_edges != st.edges) fail("nilpotent edge partition");
    return st;
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 10;
    if (maxW < 5 || maxW > 12) return 2;
    std::vector<std::vector<MateID>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 5; W <= maxW; ++W) {
        NilpotentStats ref{};
        bool first = true;
        for (int p = W - 1; p >= 3; --p) {
            const auto s = verify_position(words[W], words[W - 1], W, p, false);
            if (first) { ref = s; first = false; }
            else if (s.states != ref.states || s.components != ref.components || s.edges != ref.edges ||
                     s.correction_edges != ref.correction_edges || s.primitive_same != ref.primitive_same ||
                     s.primitive_delete_lr != ref.primitive_delete_lr ||
                     s.primitive_insert_lr_neg != ref.primitive_insert_lr_neg)
                fail("nilpotent forward position totals");
            ref.max_pairs = std::max(ref.max_pairs, s.max_pairs);
            ref.max_depth = std::max(ref.max_depth, s.max_depth);
            ref.max_in = std::max(ref.max_in, s.max_in);
            ref.max_out = std::max(ref.max_out, s.max_out);
        }
        for (int p = 1; p <= W - 3; ++p) {
            const auto s = verify_position(words[W], words[W - 1], W, p, true);
            if (s.states != ref.states || s.components != ref.components || s.edges != ref.edges ||
                s.correction_edges != ref.correction_edges || s.primitive_same != ref.primitive_same ||
                s.primitive_delete_lr != ref.primitive_delete_lr ||
                s.primitive_insert_lr_neg != ref.primitive_insert_lr_neg)
                fail("nilpotent reverse position totals");
            ref.max_pairs = std::max(ref.max_pairs, s.max_pairs);
            ref.max_depth = std::max(ref.max_depth, s.max_depth);
            ref.max_in = std::max(ref.max_in, s.max_in);
            ref.max_out = std::max(ref.max_out, s.max_out);
        }

        const Rank delta = words[W - 2].size() - words[W - 3].size();
        if (ref.primitive_delete_lr != delta || ref.primitive_insert_lr_neg != delta ||
            ref.primitive_same + 2 * delta != ref.states)
            fail("nilpotent primitive class formula");
        std::cout << "W=" << W
                  << " states=" << ref.states
                  << " components=" << ref.components
                  << " correction_edges=" << ref.correction_edges
                  << " max_pairs=" << ref.max_pairs
                  << " max_depth=" << ref.max_depth
                  << " max_in=" << ref.max_in
                  << " max_out=" << ref.max_out
                  << " primitive_same=" << ref.primitive_same
                  << " delete_LR=" << ref.primitive_delete_lr
                  << " insert_LR_neg=" << ref.primitive_insert_lr_neg
                  << " forced_matching=1 N5_zero=1 forward=OK reverse=OK\n";
    }

    constexpr Rank M28 = 385719506620ULL;
    constexpr Rank M27 = 135015505407ULL;
    constexpr Rank M26 = 47337954326ULL;
    constexpr Rank M25 = 16626415975ULL;
    constexpr Rank states = M28 + M27 - M26;
    constexpr Rank delta = M26 - M25;
    std::cout << "W=28_theory states=" << states
              << " primitive_topology_change_each_direction=" << delta
              << " topology_changing_fraction=" << (double(2 * delta) / double(states))
              << " primitive_preserving_fraction=" << (1.0 - double(2 * delta) / double(states))
              << " correction_nilpotency_candidate=N^5=0"
              << '\n';
    std::cout << "ALL_OK production_nilpotent_shear=1\n";
    return 0;
}
