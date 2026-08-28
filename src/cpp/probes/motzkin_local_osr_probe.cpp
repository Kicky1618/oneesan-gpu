#include <array>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using u64 = std::uint64_t;
static constexpr std::uint32_t P = 998244353u;

static std::uint32_t mul(std::uint32_t a, std::uint32_t b) {
    return std::uint32_t((u64(a) * b) % P);
}
static std::uint32_t pw(std::uint32_t a, u64 e) {
    std::uint32_t r = 1;
    while (e) {
        if (e & 1) r = mul(r, a);
        a = mul(a, a);
        e >>= 1;
    }
    return r;
}
static std::uint32_t inv(std::uint32_t a) { return pw(a, P - 2); }
static std::uint32_t neg(std::uint32_t a) { return a ? P - a : 0; }

using Mat = std::array<std::array<std::uint32_t, 9>, 9>;
static Mat zero() { return {}; }
static int id1(int a, int b) { return 3 * a + b; }

static Mat add(std::initializer_list<const Mat*> xs) {
    Mat z{};
    for (auto p : xs)
        for (int i = 0; i < 9; ++i)
            for (int j = 0; j < 9; ++j) {
                std::uint32_t v = z[i][j] + (*p)[i][j];
                if (v >= P) v -= P;
                z[i][j] = v;
            }
    return z;
}

static int rank_mod(std::vector<std::vector<std::uint32_t>> a) {
    int m = int(a.size()), n = m ? int(a[0].size()) : 0, r = 0;
    for (int c = 0; c < n && r < m; ++c) {
        int p = r;
        while (p < m && !a[p][c]) ++p;
        if (p == m) continue;
        std::swap(a[p], a[r]);
        auto q = inv(a[r][c]);
        for (int i = r + 1; i < m; ++i) {
            if (!a[i][c]) continue;
            auto f = mul(a[i][c], q);
            for (int j = c; j < n; ++j)
                a[i][j] = (a[i][j] + P - mul(f, a[r][j])) % P;
        }
        ++r;
    }
    return r;
}

static int matrix_rank(const Mat& a) {
    std::vector<std::vector<std::uint32_t>> v(9, std::vector<std::uint32_t>(9));
    for (int i = 0; i < 9; ++i) for (int j = 0; j < 9; ++j) v[i][j] = a[i][j];
    return rank_mod(std::move(v));
}

// Realignment: O_{a'b',ab} -> R(O)_{a'a,b'b}.  Its matrix rank is the
// operator-Schmidt rank across the two qutrit sites.
static int osr(const Mat& a) {
    std::vector<std::vector<std::uint32_t>> r(9, std::vector<std::uint32_t>(9));
    for (int ap = 0; ap < 3; ++ap)
        for (int bp = 0; bp < 3; ++bp)
            for (int aa = 0; aa < 3; ++aa)
                for (int bb = 0; bb < 3; ++bb)
                    r[id1(ap, aa)][id1(bp, bb)] = a[id1(ap, bp)][id1(aa, bb)];
    return rank_mod(std::move(r));
}

int main() {
    // labels: 0 -> +1, 1 -> 0(vacancy), 2 -> -1.
    // q = 3^((p-1)/4) is a square root of -1 modulo 998244353.
    const std::uint32_t q = pw(3, (P - 1) / 4);
    const std::uint32_t qi = inv(q);
    if (mul(q, q) != P - 1) return 2;

    Mat I = zero(), R = zero(), L = zero(), E = zero(), eps = zero();
    for (int i = 0; i < 9; ++i) I[i][i] = 1;

    // r(v_i \otimes v_j) = delta_{j,0} v_0 \otimes v_i
    // l(v_i \otimes v_j) = delta_{i,0} v_j \otimes v_0
    for (int i = 0; i < 3; ++i) {
        R[id1(1, i)][id1(i, 1)] = 1;
        L[id1(i, 1)][id1(1, i)] = 1;
    }

    // Doty--Giaquinto, plus-sign convention and alpha=1, on the weight-zero
    // basis (+,-), (0,0), (-,+).  At q=i this is the Motzkin parameter delta=1.
    const std::array<int, 3> z = {id1(0, 2), id1(1, 1), id1(2, 0)};
    const std::uint32_t B[3][3] = {
        {q, neg(q), P - 1},
        {P - 1, 1, qi},
        {P - 1, 1, qi},
    };
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            E[z[i]][z[j]] = B[i][j];

    // PTL dense generator epsilon=(1-p_i)e(1-p_i): remove vacancy row/column.
    for (int i : {0, 2})
        for (int j : {0, 2})
            eps[z[i]][z[j]] = E[z[i]][z[j]];

    struct Test { const char* name; Mat m; int want_rank, want_osr; };
    std::vector<Test> tests;
    tests.push_back({"I", I, 9, 1});
    tests.push_back({"r", R, 3, 3});
    tests.push_back({"l", L, 3, 3});
    tests.push_back({"e_full", E, 1, 9});
    tests.push_back({"epsilon_PTL", eps, 1, 4});
    tests.push_back({"I+e_full", add({&I, &E}), 9, 9});
    tests.push_back({"I+epsilon", add({&I, &eps}), 9, 5});
    tests.push_back({"I+r+l+e", add({&I, &R, &L, &E}), 7, 9});
    tests.push_back({"I+r+l+epsilon", add({&I, &R, &L, &eps}), 7, 9});

    for (auto const& t : tests) {
        int mr = matrix_rank(t.m), sr = osr(t.m);
        std::cout << t.name << " rank=" << mr << " osr=" << sr << "\n";
        if (mr != t.want_rank || sr != t.want_osr) {
            std::cerr << "MISMATCH " << t.name << "\n";
            return 1;
        }
    }

    std::cout << "generic_crossing_osr_bound_per_row=9\n";
    for (int tail = 1; tail <= 4; ++tail) {
        u64 one_layer = 1, double_layer = 1;
        for (int i = 0; i < tail; ++i) { one_layer *= 9; double_layer *= 81; }
        std::cout << "tail_rows=" << tail
                  << " mpo_bound=" << one_layer
                  << " double_layer_bound=" << double_layer << "\n";
    }
    return 0;
}
