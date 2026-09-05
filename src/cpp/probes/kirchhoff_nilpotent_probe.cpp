#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

namespace {

using u64 = std::uint64_t;
using i64 = std::int64_t;

constexpr u64 MOD = 4294967291ULL;  // largest 32-bit prime

u64 mod_pow(u64 a, u64 e) {
    u64 r = 1;
    while (e != 0) {
        if (e & 1) r = static_cast<u64>((__uint128_t)r * a % MOD);
        a = static_cast<u64>((__uint128_t)a * a % MOD);
        e >>= 1;
    }
    return r;
}

u64 det_mod(std::vector<u64> a, int n) {
    u64 det = 1;
    for (int col = 0; col < n; ++col) {
        int pivot = col;
        while (pivot < n && a[pivot * n + col] == 0) ++pivot;
        if (pivot == n) return 0;
        if (pivot != col) {
            for (int j = col; j < n; ++j) {
                std::swap(a[pivot * n + j], a[col * n + j]);
            }
            det = det == 0 ? 0 : MOD - det;
        }

        const u64 p = a[col * n + col];
        det = static_cast<u64>((__uint128_t)det * p % MOD);
        const u64 inv = mod_pow(p, MOD - 2);

        for (int row = col + 1; row < n; ++row) {
            const u64 x = a[row * n + col];
            if (x == 0) continue;
            const u64 factor = static_cast<u64>((__uint128_t)x * inv % MOD);
            for (int j = col; j < n; ++j) {
                const u64 sub = static_cast<u64>((__uint128_t)factor * a[col * n + j] % MOD);
                u64& dst = a[row * n + j];
                dst = dst >= sub ? dst - sub : dst + MOD - sub;
            }
        }
    }
    return det;
}

struct Grid {
    int cells;
    int w;
    int vertices;
    int s;
    int t;
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

// Evaluate the (s,s)-minor of M after:
//   * factoring one x_v from each row of the weighted Kirchhoff minor,
//   * taking the constant term 1 in M_ss (the r--s edge),
//   * setting x_t = 0,
//   * setting x_v in {0,1} for every v != t.
// The remaining matrix has size |V|-1 (row/column s removed).
std::vector<u64> build_eval_matrix(const Grid& g, u64 subset) {
    const int dim = g.vertices - 1;
    std::vector<int> old_to_new(g.vertices, -1);
    int next = 0;
    for (int v = 0; v < g.vertices; ++v) {
        if (v != g.s) old_to_new[v] = next++;
    }

    std::vector<int> var_bit(g.vertices, -1);
    int bit = 0;
    for (int v = 0; v < g.vertices; ++v) {
        if (v != g.t) var_bit[v] = bit++;
    }

    auto x = [&](int v) -> u64 {
        if (v == g.t) return 0;
        return (subset >> var_bit[v]) & 1ULL;
    };

    std::vector<u64> b(dim * dim, 0);
    for (int a = 0; a < g.vertices; ++a) {
        if (a == g.s) continue;
        const int row = old_to_new[a];

        u64 diag = 0;
        // Root edge r--a has weight x_a^2 for a != s,t.
        // After factoring x_a from row a, it contributes x_a.
        if (a != g.t) diag = x(a);
        for (int u : g.adj[a]) {
            diag += x(u);
            if (diag >= MOD) diag -= MOD;
        }
        b[row * dim + row] = diag;

        // Grid edge (a,b) has weight x_a x_b. After factoring x_a
        // from row a, the off-diagonal entry is -x_b.
        for (int nb : g.adj[a]) {
            if (nb == g.s) continue;  // column s was removed
            const int col = old_to_new[nb];
            const u64 xv = x(nb);
            b[row * dim + col] = xv == 0 ? 0 : MOD - xv;
        }
    }
    return b;
}

u64 kirchhoff_path_count_mod(int cells) {
    const Grid g = make_grid(cells);
    const int vars = g.vertices - 1;  // all x_v except x_t
    if (vars >= 63) {
        throw std::runtime_error("probe only supports fewer than 63 variables");
    }

    const u64 subsets = 1ULL << vars;
    u64 acc = 0;
    for (u64 mask = 0; mask < subsets; ++mask) {
        const auto mat = build_eval_matrix(g, mask);
        const u64 d = det_mod(mat, g.vertices - 1);
        const int parity = (vars - __builtin_popcountll(mask)) & 1;
        if (parity == 0) {
            acc += d;
            if (acc >= MOD) acc -= MOD;
        } else {
            acc = acc >= d ? acc - d : acc + MOD - d;
        }
    }
    return acc;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<int> ns;
    if (argc >= 2) {
        ns.push_back(std::stoi(argv[1]));
    } else {
        ns = {1, 2, 3};
    }

    for (int n : ns) {
        const auto begin = std::chrono::steady_clock::now();
        const u64 ans = kirchhoff_path_count_mod(n);
        const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        std::cout << "n=" << n << " kirchhoff_mod=" << ans << " modulus=" << MOD
                  << " seconds=" << sec << '\n';
    }
}
