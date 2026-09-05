#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using u64 = std::uint64_t;
constexpr u64 MOD = 4294967291ULL;

struct Grid {
    int cells = 0;
    int w = 0;
    int vertices = 0;
    int s = 0;
    int t = 0;
    std::vector<std::vector<int>> adj;
};

Grid make_grid(int cells) {
    Grid g;
    g.cells = cells;
    g.w = cells + 1;
    g.vertices = g.w * g.w;
    g.s = 0;
    g.t = g.vertices - 1;
    g.adj.assign(g.vertices, {});

    auto id = [&](int y, int x) { return y * g.w + x; };
    constexpr int dy[4] = {-1, 1, 0, 0};
    constexpr int dx[4] = {0, 0, -1, 1};
    for (int y = 0; y < g.w; ++y) {
        for (int x = 0; x < g.w; ++x) {
            const int v = id(y, x);
            for (int d = 0; d < 4; ++d) {
                const int ny = y + dy[d];
                const int nx = x + dx[d];
                if (0 <= ny && ny < g.w && 0 <= nx && nx < g.w) {
                    g.adj[v].push_back(id(ny, nx));
                }
            }
        }
    }
    return g;
}

struct Atom {
    int col = -1;
    int var = -1;
    int coeff = 0;  // +1 or -1
};

struct Key {
    u64 cols = 0;
    u64 vars = 0;

    bool operator==(const Key& other) const noexcept {
        return cols == other.cols && vars == other.vars;
    }
};

struct KeyHash {
    std::size_t operator()(const Key& k) const noexcept {
        u64 x = k.cols + 0x9e3779b97f4a7c15ULL;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        x ^= x >> 31;
        u64 y = k.vars + 0x517cc1b727220a95ULL;
        y = (y ^ (y >> 30)) * 0xbf58476d1ce4e5b9ULL;
        y = (y ^ (y >> 27)) * 0x94d049bb133111ebULL;
        y ^= y >> 31;
        return static_cast<std::size_t>(x ^ (y + 0x9e3779b97f4a7c15ULL + (x << 6) + (x >> 2)));
    }
};

struct Problem {
    int m = 0;
    std::vector<int> row_vertex;
    std::vector<std::vector<Atom>> atoms;
    std::vector<u64> expire_cols;
    std::vector<u64> expire_vars;
};

Problem build_problem(int cells) {
    const Grid g = make_grid(cells);
    const int m = g.vertices - 1;
    if (m >= 64) throw std::runtime_error("probe supports at most 63 columns/variables");

    std::vector<int> col_index(g.vertices, -1);
    std::vector<int> var_index(g.vertices, -1);
    int ci = 0;
    int vi = 0;
    for (int v = 0; v < g.vertices; ++v) {
        if (v != g.s) col_index[v] = ci++;
        if (v != g.t) var_index[v] = vi++;
    }

    Problem p;
    p.m = m;
    for (int a = 0; a < g.vertices; ++a) {
        if (a != g.s) p.row_vertex.push_back(a);
    }
    p.atoms.resize(m);

    for (int row = 0; row < m; ++row) {
        const int a = p.row_vertex[row];
        auto& out = p.atoms[row];

        // Diagonal root term x_a for ordinary vertices.
        if (a != g.t) {
            out.push_back({col_index[a], var_index[a], +1});
        }

        for (int nb : g.adj[a]) {
            if (nb == g.t) continue;  // x_t was set to zero

            // Diagonal contribution +x_nb.
            out.push_back({col_index[a], var_index[nb], +1});

            // Off-diagonal contribution -x_nb. Column s was deleted.
            if (nb != g.s) {
                out.push_back({col_index[nb], var_index[nb], -1});
            }
        }
    }

    std::vector<int> last_col(m, -1), last_var(m, -1);
    for (int row = 0; row < m; ++row) {
        for (const Atom& a : p.atoms[row]) {
            last_col[a.col] = std::max(last_col[a.col], row);
            last_var[a.var] = std::max(last_var[a.var], row);
        }
    }
    p.expire_cols.assign(m, 0);
    p.expire_vars.assign(m, 0);
    for (int i = 0; i < m; ++i) {
        if (last_col[i] < 0 || last_var[i] < 0) {
            throw std::runtime_error("column or variable never appears");
        }
        p.expire_cols[last_col[i]] |= 1ULL << i;
        p.expire_vars[last_var[i]] |= 1ULL << i;
    }
    return p;
}

inline void add_mod(u64& dst, u64 value) {
    dst += value;
    if (dst >= MOD) dst -= MOD;
}

inline u64 neg_mod(u64 x) {
    return x == 0 ? 0 : MOD - x;
}

struct Result {
    u64 value = 0;
    std::size_t peak_states = 0;
    std::vector<std::size_t> states_per_row;
};

Result solve(int cells) {
    const Problem p = build_problem(cells);
    std::unordered_map<Key, u64, KeyHash> cur, next;
    cur.reserve(1024);
    cur[{0, 0}] = 1;

    Result result;
    result.peak_states = 1;
    result.states_per_row.reserve(p.m);

    u64 expired_cols = 0;
    u64 expired_vars = 0;

    for (int row = 0; row < p.m; ++row) {
        const u64 must_cols = expired_cols | p.expire_cols[row];
        const u64 must_vars = expired_vars | p.expire_vars[row];
        next.clear();
        next.reserve(cur.size() * 3 + 16);

        for (const auto& [state, ways] : cur) {
            for (const Atom& a : p.atoms[row]) {
                const u64 cb = 1ULL << a.col;
                const u64 vb = 1ULL << a.var;
                if ((state.cols & cb) || (state.vars & vb)) continue;

                const u64 new_cols = state.cols | cb;
                const u64 new_vars = state.vars | vb;
                if ((new_cols & must_cols) != must_cols) continue;
                if ((new_vars & must_vars) != must_vars) continue;

                // Rows are processed in increasing order. Choosing column c adds
                // one inversion for every previously chosen column greater than c.
                const int inv_parity = __builtin_popcountll(state.cols >> (a.col + 1)) & 1;
                const bool negative = (a.coeff < 0) ^ (inv_parity != 0);
                const u64 delta = negative ? neg_mod(ways) : ways;
                add_mod(next[{new_cols, new_vars}], delta);
            }
        }

        expired_cols = must_cols;
        expired_vars = must_vars;
        cur.swap(next);
        result.states_per_row.push_back(cur.size());
        result.peak_states = std::max(result.peak_states, cur.size());
    }

    const u64 full = p.m == 64 ? ~0ULL : ((1ULL << p.m) - 1);
    const auto it = cur.find({full, full});
    result.value = it == cur.end() ? 0 : it->second;
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<int> ns;
    if (argc >= 2) {
        ns.push_back(std::stoi(argv[1]));
    } else {
        ns = {1, 2, 3, 4, 5};
    }

    for (int n : ns) {
        const auto begin = std::chrono::steady_clock::now();
        const Result r = solve(n);
        const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        std::cout << "n=" << n << " mixed_det_mod=" << r.value << " modulus=" << MOD
                  << " peak_states=" << r.peak_states << " seconds=" << sec << '\n';
        std::cout << "  row_states:";
        for (std::size_t x : r.states_per_row) std::cout << ' ' << x;
        std::cout << '\n';
    }
}
