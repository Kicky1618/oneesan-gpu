#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

using Pairing = std::vector<int>; // mate[i]

static void gen_range(int lo, int hi, Pairing &mate, std::vector<Pairing> &out) {
    if (lo > hi) {
        out.push_back(mate);
        return;
    }
    // Generate noncrossing perfect matchings recursively.  Pair lo with j.
    for (int j = lo + 1; j <= hi; j += 2) {
        std::vector<Pairing> left, right;
        Pairing ml(mate.size(), -1), mr(mate.size(), -1);
        gen_range(lo + 1, j - 1, ml, left);
        gen_range(j + 1, hi, mr, right);
        for (auto const &l : left) for (auto const &r : right) {
            Pairing x = mate;
            x[lo] = j; x[j] = lo;
            for (int k = lo + 1; k < j; ++k) if (l[k] >= 0) x[k] = l[k];
            for (int k = j + 1; k <= hi; ++k) if (r[k] >= 0) x[k] = r[k];
            out.push_back(std::move(x));
        }
    }
}

static std::vector<Pairing> all_pairings(int points) {
    if (points < 0 || (points & 1)) throw std::runtime_error("points must be even");
    if (points == 0) return {Pairing{}};
    Pairing m(points, -1);
    std::vector<Pairing> out;
    gen_range(0, points - 1, m, out);
    return out;
}

// The two noncrossing pairings are drawn on opposite sides of the cut.  Their
// union is a disjoint collection of even cycles.  For the defect formulation,
// point 0 is the extra exterior endpoint; cutting the unique cycle through it
// gives exactly one source-to-target path.  Hence point-symmetry compatibility
// is equivalent to the union being connected.
static bool compatible(Pairing const &a, Pairing const &b) {
    const int n = (int)a.size();
    if ((int)b.size() != n) return false;
    std::vector<uint8_t> seen(n, 0);
    std::vector<int> st{0}; seen[0] = 1; int got = 0;
    while (!st.empty()) {
        int u = st.back(); st.pop_back(); ++got;
        int vs[2] = {a[u], b[u]};
        for (int v : vs) if (!seen[v]) { seen[v] = 1; st.push_back(v); }
    }
    return got == n;
}

struct CSR {
    std::vector<uint64_t> row;
    std::vector<uint32_t> col;
};

static CSR build_csr(std::vector<Pairing> const &s) {
    CSR g; g.row.resize(s.size() + 1, 0);
    for (size_t i = 0; i < s.size(); ++i) {
        for (size_t j = 0; j < s.size(); ++j) {
            if (compatible(s[i], s[j])) g.col.push_back((uint32_t)j);
        }
        g.row[i + 1] = g.col.size();
    }
    return g;
}

static uint64_t bilinear_csr(CSR const &g, std::vector<uint64_t> const &x,
                             std::vector<uint64_t> const &y, uint64_t mod) {
    __uint128_t acc = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        uint64_t yi = 0;
        for (uint64_t e = g.row[i]; e < g.row[i + 1]; ++e) {
            yi += y[g.col[e]];
            if (yi >= mod) yi -= mod;
        }
        acc += (__uint128_t)x[i] * yi;
        acc %= mod;
    }
    return (uint64_t)acc;
}

int main(int argc, char **argv) {
    int m = argc > 1 ? std::atoi(argv[1]) : 8; // Catalan order; frontier occupied k=2m-1
    if (m < 1 || m > 11) {
        std::cerr << "m must be 1..11 for the explicit CSR probe\n";
        return 2;
    }
    auto states = all_pairings(2 * m);
    auto csr = build_csr(states);
    uint64_t maxdeg = 0;
    for (size_t i = 0; i < states.size(); ++i)
        maxdeg = std::max(maxdeg, csr.row[i + 1] - csr.row[i]);

    constexpr uint64_t MOD = 4294967291ULL;
    std::vector<uint64_t> x(states.size()), y(states.size());
    for (size_t i = 0; i < states.size(); ++i) {
        x[i] = (0x9e3779b97f4a7c15ULL * (i + 1)) % MOD;
        y[i] = (0xbf58476d1ce4e5b9ULL * (i + 3)) % MOD;
    }
    uint64_t z = bilinear_csr(csr, x, y, MOD);
    std::cout << "m=" << m
              << " frontier_occupied=" << (2 * m - 1)
              << " catalan_states=" << states.size()
              << " directed_matching_edges=" << csr.col.size()
              << " avg_degree=" << (double)csr.col.size() / states.size()
              << " max_degree=" << maxdeg
              << " checksum=" << z << "\n";
}
