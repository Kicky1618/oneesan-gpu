#pragma push_macro("main")
#undef main
#define main two_cell_row_turn_probe_main_unused
#include "two_cell_row_turn_probe.cpp"
#pragma pop_macro("main")

#include <deque>

namespace {

struct TurnComponentStats {
    Rank states = 0;
    Rank edges = 0;
    Rank components = 0;
    Rank max_pairs = 0;
    Rank max_cycle_rank = 0;
    Rank coeff1 = 0;
    Rank coeff2 = 0;
};

TurnComponentStats verify_right_turn_components(
    int W,
    const std::vector<std::vector<Word>>& words
) {
    const auto src = q_basis(W, W - 3, words);
    const auto dst = reverse_q_basis(W, W - 2, words);
    if (src.size() != dst.size()) fail("turn component non-square");
    const Rank n = src.size();

    std::map<Key, Rank> src_rank, dst_rank;
    for (Rank r = 0; r < n; ++r) {
        if (!src_rank.emplace(src[r], r).second || !dst_rank.emplace(dst[r], r).second)
            fail("turn component duplicate rank");
    }

    std::vector<std::vector<Rank>> adj(static_cast<std::size_t>(2 * n));
    TurnComponentStats st;
    st.states = n;
    for (Rank s = 0; s < n; ++s) {
        const CVec col = turn_right_basis(src[s], W);
        for (const auto& [k, c] : col) {
            if (c < 1 || c > 2) fail("turn component coefficient");
            const auto it = dst_rank.find(k);
            if (it == dst_rank.end()) fail("turn component destination");
            const Rank d = it->second;
            adj[d].push_back(n + s);
            adj[n + s].push_back(d);
            ++st.edges;
            if (c == 1) ++st.coeff1;
            else ++st.coeff2;
        }
    }

    std::vector<std::int64_t> component(static_cast<std::size_t>(2 * n), -1);
    for (Rank root = 0; root < 2 * n; ++root) {
        if (component[root] >= 0) continue;
        const std::int64_t cid = static_cast<std::int64_t>(st.components++);
        std::vector<Rank> stack{root};
        component[root] = cid;
        Rank vertices = 0, rows = 0, cols = 0, degree_sum = 0;
        while (!stack.empty()) {
            const Rank v = stack.back();
            stack.pop_back();
            ++vertices;
            rows += v < n;
            cols += v >= n;
            degree_sum += adj[v].size();
            for (Rank z : adj[v]) {
                if (component[z] >= 0) continue;
                component[z] = cid;
                stack.push_back(z);
            }
        }
        if (rows != cols) fail("turn component imbalance W=" + std::to_string(W));
        const Rank e = degree_sum / 2;
        const Rank cycle_rank = e + 1 - vertices;
        st.max_cycle_rank = std::max(st.max_cycle_rank, cycle_rank);
        st.max_pairs = std::max(st.max_pairs, rows);
    }

    const Rank m1 = words[W - 1].size();
    const Rank m2 = words[W - 2].size();
    const Rank m3 = words[W - 3].size();
    const Rank expected_edges = 3 * (m1 - m3);
    if (st.components != m2 || st.edges != expected_edges ||
        st.coeff1 != 2 * (m1 - m3) || st.coeff2 != m1 - m3)
        fail("turn component formula W=" + std::to_string(W));

    // The same unrestricted C-word seed used by the interior forest labels
    // every physical row-turn component exactly once.
    std::vector<std::uint8_t> seed_seen(static_cast<std::size_t>(st.components));
    for (const Word& u : words[W - 2]) {
        const Key seed = project_key(Key{'C', u}, W - 3, W);
        const auto it = src_rank.find(seed);
        if (it == src_rank.end()) fail("turn seed outside source layout");
        const std::int64_t cid = component[n + it->second];
        if (cid < 0 || seed_seen[static_cast<std::size_t>(cid)])
            fail("turn seed collision W=" + std::to_string(W));
        seed_seen[static_cast<std::size_t>(cid)] = 1;
    }
    for (std::uint8_t x : seed_seen)
        if (!x) fail("turn unseeded component W=" + std::to_string(W));

    return st;
}

Rank turn_count_words(int W) {
    std::vector<Rank> cur(static_cast<std::size_t>(W + 2));
    std::vector<Rank> nxt(static_cast<std::size_t>(W + 2));
    cur[1] = 1;
    for (int pos = 0; pos < W; ++pos) {
        std::fill(nxt.begin(), nxt.end(), 0);
        for (int h = 0; h <= W; ++h) {
            const Rank x = cur[h];
            if (!x) continue;
            nxt[h] += x;
            if (h > 0) nxt[h - 1] += x;
            nxt[h + 1] += x;
        }
        cur.swap(nxt);
    }
    return cur[0];
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 12;
    if (maxW < 4 || maxW > 15) return 2;

    std::vector<std::vector<Word>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 4; W <= maxW; ++W) {
        const TurnComponentStats s = verify_right_turn_components(W, words);
        std::cout << "W=" << W
                  << " states=" << s.states
                  << " components=" << s.components
                  << " avg_pairs=" << double(s.states) / double(s.components)
                  << " max_pairs=" << s.max_pairs
                  << " edges=" << s.edges
                  << " coeff1=" << s.coeff1
                  << " coeff2=" << s.coeff2
                  << " max_cycle_rank=" << s.max_cycle_rank
                  << " component_seed_MWm2=1"
                  << " OK\n";
    }

    const Rank m27 = turn_count_words(27);
    const Rank m26 = turn_count_words(26);
    const Rank m25 = turn_count_words(25);
    const Rank n28 = m27 + m26 - m25;
    const Rank edges28 = 3 * (m27 - m25);
    const Rank saved_loads28 = edges28 - n28;
    const double saved_gib28 = double(saved_loads28) * 4.0 / double(1ull << 30);
    std::cout << "W=28_theory states=" << n28
              << " components=" << m26
              << " turn_formula_gather_loads=" << edges28
              << " component_kernel_loads=" << n28
              << " saved_value_loads=" << saved_loads28
              << " saved_u32_gib_per_turn=" << saved_gib28
              << "\n";
    std::cout << "ALL_OK physical_turn_component_partition=1"
              << " component_table_bytes=0\n";
    return 0;
}
