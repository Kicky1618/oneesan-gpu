#pragma push_macro("main")
#undef main
#define main two_cell_channel_probe_main_unused
#include "two_cell_channel_probe.cpp"
#pragma pop_macro("main")

#include <deque>
#include <limits>

namespace {

struct ForestStats {
    Rank n = 0;
    Rank edges = 0;
    Rank components = 0;
    Rank max_pairs = 0;
    Rank max_row_degree = 0;
    Rank max_col_degree = 0;
    Rank lifting_adds = 0;
};

ForestStats verify_forest_step(
    int W,
    int i,
    const std::vector<std::vector<Word>>& words
) {
    const ReducedLayout src = make_layout(W, i, words);
    const ReducedLayout dst = make_layout(W, i + 1, words);
    if (src.size() != dst.size()) fail("forest non-square reduced step");

    const Rank n = src.size();
    const Rank bad = std::numeric_limits<Rank>::max();
    std::vector<std::vector<Rank>> adj(static_cast<std::size_t>(2 * n));
    std::vector<std::pair<Rank, Rank>> edge;

    for (Rank s = 0; s < n; ++s) {
        const CVec col = K_basis(src.key[s], W, i);
        if (col.empty() || col.size() > 3)
            fail("forest source fanout W=" + std::to_string(W));
        for (const auto& [k, c] : col) {
            if (c != 1) fail("forest nonunit coefficient W=" + std::to_string(W));
            const auto it = dst.rank.find(k);
            if (it == dst.rank.end()) fail("forest destination outside layout");
            const Rank d = it->second;
            adj[d].push_back(n + s);
            adj[n + s].push_back(d);
            edge.emplace_back(d, s);
        }
    }

    ForestStats st;
    st.n = n;
    st.edges = edge.size();
    for (Rank d = 0; d < n; ++d)
        st.max_row_degree = std::max<Rank>(st.max_row_degree, adj[d].size());
    for (Rank s = 0; s < n; ++s)
        st.max_col_degree = std::max<Rank>(st.max_col_degree, adj[n + s].size());

    std::vector<std::int64_t> component(static_cast<std::size_t>(2 * n), -1);
    std::vector<std::vector<Rank>> component_vertices;
    for (Rank root = 0; root < 2 * n; ++root) {
        if (component[root] >= 0) continue;
        const std::int64_t cid = static_cast<std::int64_t>(component_vertices.size());
        std::vector<Rank> vertices;
        std::vector<Rank> stack{root};
        component[root] = cid;
        Rank rows = 0, cols = 0, degree_sum = 0;
        while (!stack.empty()) {
            const Rank v = stack.back();
            stack.pop_back();
            vertices.push_back(v);
            rows += v < n;
            cols += v >= n;
            degree_sum += adj[v].size();
            for (Rank z : adj[v]) {
                if (component[z] >= 0) continue;
                component[z] = cid;
                stack.push_back(z);
            }
        }
        if (rows != cols)
            fail("forest unbalanced component W=" + std::to_string(W));
        const Rank e = degree_sum / 2;
        if (e + 1 != vertices.size())
            fail("forest cycle W=" + std::to_string(W) + " i=" + std::to_string(i));
        st.max_pairs = std::max(st.max_pairs, rows);
        component_vertices.push_back(std::move(vertices));
    }
    st.components = component_vertices.size();

    const Rank m2 = words[W - 2].size();
    const Rank expected_edges = 2 * n - m2;
    if (st.components != m2 || st.edges != expected_edges)
        fail("forest component/edge formula W=" + std::to_string(W));

    // There is exactly one canonical pre-quotient C_i representative per
    // source component.  Hence components can be enumerated by the ordinary
    // M_{W-2} words without storing a component table.
    std::vector<std::uint8_t> seed_seen(static_cast<std::size_t>(st.components));
    for (const Word& u : words[W - 2]) {
        const Key seed = project_key(Key{'C', u}, i, W);
        const auto it = src.rank.find(seed);
        if (it == src.rank.end()) fail("forest seed outside source layout");
        const std::int64_t cid = component[n + it->second];
        if (cid < 0 || seed_seen[static_cast<std::size_t>(cid)])
            fail("forest seed collision W=" + std::to_string(W));
        seed_seen[static_cast<std::size_t>(cid)] = 1;
    }
    for (std::uint8_t x : seed_seen)
        if (!x) fail("forest unseeded component W=" + std::to_string(W));

    // A forest has at most one perfect matching.  Leaf peeling proves that
    // every component has one and constructs the row<->column permutation.
    std::vector<Rank> matched_col_of_row(static_cast<std::size_t>(n), bad);
    std::vector<Rank> matched_row_of_col(static_cast<std::size_t>(n), bad);
    for (const auto& vertices : component_vertices) {
        std::vector<std::uint8_t> active(vertices.size(), 1);
        std::vector<Rank> degree(vertices.size());
        std::map<Rank, Rank> local;
        for (Rank q = 0; q < vertices.size(); ++q) {
            local.emplace(vertices[q], q);
            degree[q] = adj[vertices[q]].size();
        }
        std::deque<Rank> leaves;
        for (Rank q = 0; q < vertices.size(); ++q)
            if (degree[q] == 1) leaves.push_back(q);

        Rank remaining = vertices.size();
        while (!leaves.empty()) {
            const Rank lv = leaves.front();
            leaves.pop_front();
            if (!active[lv] || degree[lv] != 1) continue;
            const Rank v = vertices[lv];
            Rank u = bad;
            Rank lu = bad;
            for (Rank z : adj[v]) {
                const Rank q = local.at(z);
                if (active[q]) {
                    u = z;
                    lu = q;
                    break;
                }
            }
            if (u == bad) fail("forest leaf without neighbor");

            const Rank row = v < n ? v : u;
            const Rank col = v < n ? u - n : v - n;
            if (row >= n || col >= n ||
                matched_col_of_row[row] != bad || matched_row_of_col[col] != bad)
                fail("forest invalid matching");
            matched_col_of_row[row] = col;
            matched_row_of_col[col] = row;

            active[lv] = active[lu] = 0;
            remaining -= 2;
            for (Rank z : adj[u]) {
                const Rank q = local.at(z);
                if (!active[q]) continue;
                if (degree[q] == 0) fail("forest degree underflow");
                --degree[q];
                if (degree[q] == 1) leaves.push_back(q);
            }
            for (Rank z : adj[v]) {
                const Rank q = local.at(z);
                if (!active[q]) continue;
                if (degree[q] == 0) fail("forest degree underflow");
                --degree[q];
                if (degree[q] == 1) leaves.push_back(q);
            }
            degree[lv] = degree[lu] = 0;
        }
        if (remaining != 0)
            fail("forest component lacks perfect matching W=" + std::to_string(W));
    }
    for (Rank r = 0; r < n; ++r)
        if (matched_col_of_row[r] == bad) fail("forest unmatched row");
    for (Rank c = 0; c < n; ++c)
        if (matched_row_of_col[c] == bad) fail("forest unmatched column");

    // Contract matching edges.  Store destination row r in the physical slot
    // of its matched source column.  The remaining edges form an oriented
    // forest.  Processing target slots before their source slots is therefore
    // an exact in-place lifting: one addition per non-matching edge.
    std::vector<std::vector<Rank>> incoming(static_cast<std::size_t>(n));
    std::vector<std::vector<Rank>> precedence(static_cast<std::size_t>(n));
    std::vector<Rank> indegree(static_cast<std::size_t>(n));
    Rank nonmatching = 0;
    for (const auto& [row, col] : edge) {
        const Rank target = matched_col_of_row[row];
        if (target == col) continue;
        incoming[target].push_back(col);      // x[col] contributes to y[target]
        precedence[target].push_back(col);    // target must execute before col
        ++indegree[col];
        ++nonmatching;
    }
    if (nonmatching != n - st.components)
        fail("forest lifting edge formula W=" + std::to_string(W));
    st.lifting_adds = nonmatching;

    std::deque<Rank> ready;
    for (Rank v = 0; v < n; ++v)
        if (indegree[v] == 0) ready.push_back(v);
    std::vector<Rank> order;
    order.reserve(static_cast<std::size_t>(n));
    while (!ready.empty()) {
        const Rank v = ready.front();
        ready.pop_front();
        order.push_back(v);
        for (Rank z : precedence[v]) {
            if (indegree[z] == 0) fail("forest precedence underflow");
            if (--indegree[z] == 0) ready.push_back(z);
        }
    }
    if (order.size() != n) fail("forest contracted orientation cycle");

    std::vector<std::uint64_t> input(static_cast<std::size_t>(n));
    std::vector<std::uint64_t> reference(static_cast<std::size_t>(n));
    for (Rank s = 0; s < n; ++s)
        input[s] = 1 + ((s * 0x9e3779b97f4a7c15ULL) ^
                        (Rank(W) << 32) ^ Rank(i));
    for (const auto& [row, col] : edge) reference[row] += input[col];

    std::vector<std::uint64_t> work = input;
    for (Rank target : order)
        for (Rank source : incoming[target])
            work[target] += work[source];
    for (Rank slot = 0; slot < n; ++slot) {
        const Rank row = matched_row_of_col[slot];
        if (work[slot] != reference[row])
            fail("forest in-place lifting mismatch W=" + std::to_string(W) +
                 " i=" + std::to_string(i));
    }

    return st;
}

Rank count_words_dp(int W) {
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
    if (maxW < 4 || maxW > 15) {
        std::cerr << "maxW must be 4..15\n";
        return 2;
    }

    std::vector<std::vector<Word>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 4; W <= maxW; ++W) {
        ForestStats worst;
        for (int i = 0; i <= W - 4; ++i) {
            const ForestStats s = verify_forest_step(W, i, words);
            if (!worst.n) worst = s;
            if (s.n != worst.n || s.edges != worst.edges ||
                s.components != worst.components || s.lifting_adds != worst.lifting_adds)
                fail("forest position-dependent global counts W=" + std::to_string(W));
            worst.max_pairs = std::max(worst.max_pairs, s.max_pairs);
            worst.max_row_degree = std::max(worst.max_row_degree, s.max_row_degree);
            worst.max_col_degree = std::max(worst.max_col_degree, s.max_col_degree);
        }
        std::cout << "W=" << W
                  << " reduced=" << worst.n
                  << " edges=" << worst.edges
                  << " components=" << worst.components
                  << " avg_pairs=" << double(worst.n) / double(worst.components)
                  << " max_pairs=" << worst.max_pairs
                  << " max_row_degree=" << worst.max_row_degree
                  << " max_col_degree=" << worst.max_col_degree
                  << " lifting_adds=" << worst.lifting_adds
                  << " forest=1 unique_matching=1 inplace_lifting=1"
                  << " OK\n";
    }

    const Rank m27 = count_words_dp(27);
    const Rank m26 = count_words_dp(26);
    const Rank m25 = count_words_dp(25);
    const Rank n28 = m27 + m26 - m25;
    const Rank edges28 = 2 * n28 - m26;
    const Rank adds28 = n28 - m26;
    std::cout << "W=28_theory reduced=" << n28
              << " components=" << m26
              << " edges=" << edges28
              << " lifting_adds=" << adds28
              << " avg_pairs=" << double(n28) / double(m26)
              << " avg_component_vertices=" << 2.0 * double(n28) / double(m26)
              << "\n";
    std::cout << "ALL_OK two_cell_support_forest=1 component_seed_MWm2=1"
              << " unique_perfect_matching=1 inplace_lifting=1\n";
    return 0;
}
