#include <cassert>
#include <cstdint>
#include <iostream>
#include <map>
#include <string>
#include <vector>

// Enumerate the colored-Motzkin model whose bivariate generating function is
//   1 / (1 - (1+y)x - 2x^2/(1-x-x^2/(1-x-x^2/(...))))
// and compare its coefficient triangle against A111960.
//
// Rules:
//   U : height +1, one color
//   D : height -1; if it lands on height 0, two colors, otherwise one
//   H : same height; at height 0 it has two colors, one ordinary and one marked
//       (the marked ground-H increments k); above height 0 it has one color.
//
// T(n,k) counts length-n paths ending at height 0 with k marked ground-H steps.

using U64 = std::uint64_t;

struct Key {
    int h = 0;
    int k = 0;
    bool operator<(Key const& o) const { return h != o.h ? h < o.h : k < o.k; }
};

static std::vector<U64> enumerate_triangle(int nmax) {
    std::map<Key, U64> cur, nxt;
    cur[{0,0}] = 1;
    std::vector<U64> flat;
    for (int n = 0; n <= nmax; ++n) {
        for (int k = 0; k <= n; ++k) flat.push_back(cur[{0,k}]);
        if (n == nmax) break;
        nxt.clear();
        for (auto const& [q, v] : cur) {
            // U
            nxt[{q.h + 1, q.k}] += v;
            // H
            if (q.h == 0) {
                nxt[{0, q.k}] += v;       // ordinary ground H
                nxt[{0, q.k + 1}] += v;   // marked ground H
            } else {
                nxt[{q.h, q.k}] += v;
            }
            // D
            if (q.h > 0) {
                if (q.h == 1) nxt[{0, q.k}] += 2 * v; // two colors on return to ground
                else nxt[{q.h - 1, q.k}] += v;
            }
        }
        cur.swap(nxt);
    }
    return flat;
}

static std::vector<std::vector<U64>> enumerate_rows(int nmax) {
    std::map<Key, U64> cur, nxt;
    cur[{0,0}] = 1;
    std::vector<std::vector<U64>> rows;
    for (int n = 0; n <= nmax; ++n) {
        std::vector<U64> row(n + 1);
        for (int k = 0; k <= n; ++k) row[k] = cur[{0,k}];
        rows.push_back(row);
        if (n == nmax) break;
        nxt.clear();
        for (auto const& [q, v] : cur) {
            nxt[{q.h + 1, q.k}] += v;
            if (q.h == 0) {
                nxt[{0,q.k}] += v;
                nxt[{0,q.k+1}] += v;
            } else nxt[{q.h,q.k}] += v;
            if (q.h > 0) {
                if (q.h == 1) nxt[{0,q.k}] += 2*v;
                else nxt[{q.h-1,q.k}] += v;
            }
        }
        cur.swap(nxt);
    }
    return rows;
}

// A111960 via the coefficient recurrence
// C(n,k)=[x^(n-k)](1-2x-3x^2)^(-(k+1)/2).
static std::vector<std::vector<U64>> renewal_rows(int nmax) {
    std::vector<std::vector<U64>> C(nmax + 1);
    for (int n = 0; n <= nmax; ++n) C[n].assign(n + 1, 0);
    // Generate by the production matrix.  If row r is c_h, then
    // next[0] = c_0 + sum_{m>=1} 2(-1)^(m-1)Cat_{m-1} c_{2m-1}
    // next[h] = c_{h-1}+c_h+sum_{m>=1}2(-1)^(m-1)Cat_{m-1}c_{h+2m-1}.
    // For integer counting we avoid signed overflow by instead use the closed
    // positive coefficient recurrence in n for each k.
    C[0][0] = 1;
    for (int k = 0; k <= nmax; ++k) {
        // a_j = [x^j](1-2x-3x^2)^(-(k+1)/2), j=n-k.
        // (j+1)a_{j+1}=(2j+k+1)a_j+3(j+k)a_{j-1}.
        std::vector<__uint128_t> a(nmax - k + 1);
        a[0] = 1;
        for (int j = 0; j < nmax - k; ++j) {
            __uint128_t rhs = __uint128_t(2*j + k + 1) * a[j];
            if (j > 0) rhs += __uint128_t(3*(j+k)) * a[j-1];
            assert(rhs % (j+1) == 0);
            a[j+1] = rhs / (j+1);
        }
        for (int n = k; n <= nmax; ++n) C[n][k] = U64(a[n-k]);
    }
    return C;
}

int main(int argc, char** argv) {
    int nmax = argc > 1 ? std::atoi(argv[1]) : 14;
    if (nmax < 0 || nmax > 24) {
        std::cerr << "nmax must be 0..24 for uint64 probe\n";
        return 2;
    }
    auto motz = enumerate_rows(nmax);
    auto ren = renewal_rows(nmax);
    for (int n = 0; n <= nmax; ++n) {
        if (motz[n] != ren[n]) {
            std::cerr << "MISMATCH n=" << n << "\n";
            std::cerr << "motz:";
            for (auto x : motz[n]) std::cerr << ' ' << x;
            std::cerr << "\nrenew:";
            for (auto x : ren[n]) std::cerr << ' ' << x;
            std::cerr << "\n";
            return 1;
        }
        U64 sum = 0;
        std::cout << "n=" << n << " :";
        for (auto x : motz[n]) { std::cout << ' ' << x; sum += x; }
        std::cout << "  sum=" << sum << "\n";
    }
    std::cout << "PASS colored Motzkin triangle = A111960 through n=" << nmax << "\n";
    return 0;
}
