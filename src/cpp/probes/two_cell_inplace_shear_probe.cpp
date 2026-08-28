#pragma push_macro("main")
#undef main
#define main two_cell_channel_probe_main_unused
#include "two_cell_channel_probe.cpp"
#pragma pop_macro("main")

#include <deque>
#include <limits>

namespace {

using EdgeList = std::vector<std::vector<Rank>>;

struct StepGraph {
    EdgeList out;
    EdgeList in;
};

StepGraph build_step_graph(
    const ReducedLayout& src,
    const ReducedLayout& dst,
    int W,
    int i
) {
    StepGraph g;
    g.out.resize(static_cast<std::size_t>(src.size()));
    g.in.resize(static_cast<std::size_t>(dst.size()));
    for (Rank s = 0; s < src.size(); ++s) {
        for (const auto& [k, c] : K_basis(src.key[s], W, i)) {
            if (c != 1) fail("shear nonunit coefficient");
            const auto it = dst.rank.find(k);
            if (it == dst.rank.end()) fail("shear destination missing");
            g.out[s].push_back(it->second);
            g.in[it->second].push_back(s);
        }
    }
    return g;
}

std::pair<std::vector<Rank>, std::vector<Rank>> forced_matching(const StepGraph& g) {
    const Rank n = static_cast<Rank>(g.out.size());
    const Rank none = std::numeric_limits<Rank>::max();
    std::vector<Rank> ldeg(n), rdeg(n), ml(n, none), mr(n, none);
    std::vector<std::uint8_t> live_l(n, 1), live_r(n, 1);
    struct Vertex { bool right; Rank id; };
    std::deque<Vertex> q;
    for (Rank u = 0; u < n; ++u) {
        ldeg[u] = static_cast<Rank>(g.out[u].size());
        rdeg[u] = static_cast<Rank>(g.in[u].size());
        if (ldeg[u] == 1) q.push_back({false, u});
        if (rdeg[u] == 1) q.push_back({true, u});
    }

    auto only_r = [&](Rank u) {
        for (Rank v : g.out[u]) if (live_r[v]) return v;
        return none;
    };
    auto only_l = [&](Rank v) {
        for (Rank u : g.in[v]) if (live_l[u]) return u;
        return none;
    };

    Rank matched = 0;
    while (!q.empty()) {
        const Vertex x = q.front();
        q.pop_front();
        Rank u = none, v = none;
        if (x.right) {
            v = x.id;
            if (!live_r[v] || rdeg[v] != 1) continue;
            u = only_l(v);
        } else {
            u = x.id;
            if (!live_l[u] || ldeg[u] != 1) continue;
            v = only_r(u);
        }
        if (u == none || v == none || !live_l[u] || !live_r[v]) continue;
        ml[u] = v;
        mr[v] = u;
        live_l[u] = live_r[v] = 0;
        ++matched;
        for (Rank vv : g.out[u]) if (live_r[vv]) {
            if (!rdeg[vv]) fail("shear right degree underflow");
            if (--rdeg[vv] == 1) q.push_back({true, vv});
        }
        for (Rank uu : g.in[v]) if (live_l[uu]) {
            if (!ldeg[uu]) fail("shear left degree underflow");
            if (--ldeg[uu] == 1) q.push_back({false, uu});
        }
    }
    if (matched != n) fail("shear matching not fully forced");
    return {std::move(ml), std::move(mr)};
}

void verify_inplace_shear(
    int W,
    int i,
    const std::vector<std::vector<Word>>& words,
    Rank& max_extra_indeg,
    Rank& max_extra_outdeg
) {
    const ReducedLayout src = make_layout(W, i, words);
    const ReducedLayout dst = make_layout(W, i + 1, words);
    const Rank n = src.size();
    const StepGraph g = build_step_graph(src, dst, W, i);
    const auto [match_l, match_r] = forced_matching(g);

    EdgeList extra_out(static_cast<std::size_t>(n));
    EdgeList extra_in(static_cast<std::size_t>(n));
    std::vector<Rank> indeg(static_cast<std::size_t>(n), 0);
    Rank extras = 0;
    for (Rank s = 0; s < n; ++s) {
        for (Rank d : g.out[s]) {
            const Rank owner = match_r[d];
            if (owner == s) {
                if (match_l[s] != d) fail("shear diagonal matching mismatch");
                continue;
            }
            extra_out[s].push_back(owner);
            extra_in[owner].push_back(s);
            ++indeg[owner];
            ++extras;
        }
    }

    const Rank m2 = static_cast<Rank>(words[W - 2].size());
    if (extras != n - m2) fail("shear extra edge count is not dim-M_{W-2}");
    for (Rank u = 0; u < n; ++u) {
        max_extra_indeg = std::max(max_extra_indeg, static_cast<Rank>(extra_in[u].size()));
        max_extra_outdeg = std::max(max_extra_outdeg, static_cast<Rank>(extra_out[u].size()));
    }

    std::deque<Rank> q;
    for (Rank u = 0; u < n; ++u) if (!indeg[u]) q.push_back(u);
    std::vector<Rank> topo;
    topo.reserve(static_cast<std::size_t>(n));
    while (!q.empty()) {
        const Rank u = q.front();
        q.pop_front();
        topo.push_back(u);
        for (Rank v : extra_out[u]) {
            if (!indeg[v]) fail("shear indegree underflow");
            if (--indeg[v] == 0) q.push_back(v);
        }
    }
    if (topo.size() != static_cast<std::size_t>(n))
        fail("matched reduced operator has cyclic correction graph");

    std::vector<std::uint64_t> value(static_cast<std::size_t>(n));
    std::vector<std::uint64_t> direct(static_cast<std::size_t>(n), 0);
    for (Rank s = 0; s < n; ++s) {
        value[s] = 1 + ((s * 0x9e3779b97f4a7c15ULL) ^
                        (std::uint64_t(W) << 40) ^ (std::uint64_t(i) << 32));
        for (Rank d : g.out[s]) direct[d] += value[s];
    }

    std::vector<std::uint64_t> expect(static_cast<std::size_t>(n));
    for (Rank u = 0; u < n; ++u) expect[u] = direct[match_l[u]];

    // P^{-1}K = I + N. Since the correction graph is a DAG, reverse
    // topological order keeps every source value untouched until all of its
    // outgoing contributions have consumed it. One in-place add per extra
    // edge is therefore sufficient.
    std::vector<std::uint64_t> inplace = value;
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        const Rank dst_owner = *it;
        for (Rank src_owner : extra_in[dst_owner])
            inplace[dst_owner] += inplace[src_owner];
    }
    if (inplace != expect) fail("in-place shear differs from reduced operator");
}

std::uint64_t count_words_u64(int W) {
    std::vector<std::uint64_t> cur(static_cast<std::size_t>(W + 2), 0), next(cur.size());
    cur[1] = 1;
    for (int p = 0; p < W; ++p) {
        std::fill(next.begin(), next.end(), 0);
        for (int h = 0; h <= W; ++h) if (cur[h]) {
            next[h] += cur[h];
            next[h + 1] += cur[h];
            if (h) next[h - 1] += cur[h];
        }
        cur.swap(next);
    }
    return cur[0];
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 14;
    if (maxW < 4 || maxW > 14) return 2;
    std::vector<std::vector<Word>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 4; W <= maxW; ++W) {
        Rank max_in = 0, max_out = 0;
        for (int i = 0; i <= W - 4; ++i)
            verify_inplace_shear(W, i, words, max_in, max_out);
        const Rank m1 = static_cast<Rank>(words[W - 1].size());
        const Rank m2 = static_cast<Rank>(words[W - 2].size());
        const Rank m3 = static_cast<Rank>(words[W - 3].size());
        const Rank dim = m1 + m2 - m3;
        const Rank extras = dim - m2;
        std::cout << "W=" << W
                  << " dim=" << dim
                  << " permutation=" << dim
                  << " shear_adds=" << extras
                  << " adds_per_state=" << (double(extras) / double(dim))
                  << " max_extra_indeg=" << max_in
                  << " max_extra_outdeg=" << max_out
                  << " inplace_exact=1\n";
    }

    constexpr int W = 28;
    const std::uint64_t m1 = count_words_u64(W - 1);
    const std::uint64_t m2 = count_words_u64(W - 2);
    const std::uint64_t m3 = count_words_u64(W - 3);
    const std::uint64_t dim = m1 + m2 - m3;
    const std::uint64_t extras = dim - m2;
    std::cout << "W28_shear_projection"
              << " reduced=" << dim
              << " permutation_edges=" << dim
              << " shear_adds=" << extras
              << " adds_per_state=" << (double(extras) / double(dim))
              << " logical_transform=P*(I+forest_DAG)"
              << '\n';
    return 0;
}
