#include <cassert>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#define main odd_tl_factorization_probe_main
#include "odd_tl_gram_factorization_probe.cpp"
#undef main

static std::uint32_t subm(std::uint32_t a, std::uint32_t b) {
    return a >= b ? a - b : std::uint32_t(u64(a) + MOD - b);
}

// Inverse of transform() in odd_tl_gram_factorization_probe.cpp.
// Forward order is local triangular shear followed by four child transforms;
// inverse order is therefore child inverses followed by the inverse shear.
static void inverse_transform(int n, int d, std::vector<std::uint32_t>& v) {
    if (n <= 1) return;
    int da = int(basis(n - 2, d - 2).words.size());
    int d0 = int(basis(n - 2, d).words.size());
    int dp = int(basis(n - 2, d + 2).words.size());
    assert(int(v.size()) == da + 2 * d0 + dp);

    std::vector<std::uint32_t> A(v.begin(), v.begin() + da);
    std::vector<std::uint32_t> B(v.begin() + da, v.begin() + da + d0);
    std::vector<std::uint32_t> C(v.begin() + da + d0, v.begin() + da + 2 * d0);
    std::vector<std::uint32_t> D(v.begin() + da + 2 * d0, v.end());

    if (da) inverse_transform(n - 2, d - 2, A);
    if (d0) {
        inverse_transform(n - 2, d, B);
        inverse_transform(n - 2, d, C);
    }
    if (dp) inverse_transform(n - 2, d + 2, D);

    // C0 = C' - J D, B0 = B' - partial D.
    if (d0 && dp) {
        std::vector<std::uint32_t> t(d0);
        apply_J(n - 2, d + 2, D, t);
        for (int i = 0; i < d0; ++i) C[i] = subm(C[i], t[i]);
        std::fill(t.begin(), t.end(), 0);
        apply_partial(n - 2, d + 2, D, t);
        for (int i = 0; i < d0; ++i) B[i] = subm(B[i], t[i]);
    }

    // Forward A used the old C, so use the already recovered C0 here.
    if (da) {
        std::vector<std::uint32_t> t(da);
        apply_partial(n - 2, d, C, t);
        for (int i = 0; i < da; ++i) A[i] = subm(A[i], t[i]);
        std::fill(t.begin(), t.end(), 0);
        apply_Q(n - 2, d, D, t);
        for (int i = 0; i < da; ++i) A[i] = subm(A[i], t[i]);
    }

    int off = 0;
    for (auto x : A) v[off++] = x;
    for (auto x : B) v[off++] = x;
    for (auto x : C) v[off++] = x;
    for (auto x : D) v[off++] = x;
}

static std::uint32_t reflected_word(std::uint32_t w, int n) {
    auto m = mates(w, n);
    std::vector<int> rm(n, -1);
    for (int i = 0; i < n; ++i)
        if (m[i] >= 0) rm[n - 1 - i] = n - 1 - m[i];

    std::uint32_t out = 0;
    for (int i = 0; i < n; ++i)
        if (rm[i] < 0 || i < rm[i]) out |= 1u << i;
    return out;
}

static void reflect_diagram_vector(int n, int d,
                                   std::vector<std::uint32_t>& v) {
    auto const& B = basis(n, d);
    std::vector<std::uint32_t> z(v.size());
    for (int i = 0; i < int(B.words.size()); ++i) {
        auto rw = reflected_word(B.words[i], n);
        auto it = B.rank.find(rw);
        assert(it != B.rank.end());
        z[it->second] = v[i];
    }
    v.swap(z);
}

static std::uint32_t explicit_reflection_bilinear(
    int n, int d,
    std::vector<std::uint32_t> const& a,
    std::vector<std::uint32_t> const& b
) {
    auto rb = b;
    reflect_diagram_vector(n, d, rb);
    return explicit_bilinear(n, d, a, rb);
}

// A004148, indexed from 0.  The observed support of the dense one-defect
// fusion reflection closure K_n for odd n is exactly a(n) through n=15.
static constexpr std::uint64_t RNA[] = {
    1,1,1,2,4,8,17,37,82,185,423,978,2283,5373,12735,30372,
    72832,175502,424748,1032004,2516347,6155441,15101701,37150472,
    91618049,226460893,560954047,1392251012ULL
};

static std::uint64_t support_nnz(int n) {
    int dim = int(basis(n, 1).words.size());
    std::uint64_t total = 0;
    for (int j = 0; j < dim; ++j) {
        std::vector<std::uint32_t> x(dim);
        x[j] = 1;
        inverse_transform(n, 1, x);
        reflect_diagram_vector(n, 1, x);
        transform(n, 1, x);
        for (auto z : x) total += (z != 0);
    }
    return total;
}

int main(int argc, char** argv) {
    int max_n = argc > 1 ? std::atoi(argv[1]) : 13;
    if (max_n > 15) max_n = 15;
    if (!(max_n & 1)) --max_n;

    std::mt19937_64 rng(0x51f15eULL);
    for (int n = 1; n <= max_n; n += 2) {
        int dim = int(basis(n, 1).words.size());

        // Verify T^{-1} and the midpoint identity on arbitrary fusion-space
        // vectors, not only vectors reached by Grid-FP.
        for (int trial = 0; trial < 4; ++trial) {
            std::vector<std::uint32_t> x(dim), y(dim);
            for (auto& z : x) z = std::uint32_t(rng() % MOD);
            for (auto& z : y) z = std::uint32_t(rng() % MOD);

            auto raw_x = x;
            auto raw_y = y;
            inverse_transform(n, 1, raw_x);
            inverse_transform(n, 1, raw_y);

            auto want = explicit_reflection_bilinear(n, 1, raw_x, raw_y);

            // Only the right member of a reflected occupancy-mask pair needs
            // the roundtrip.  The left member stays in fusion coordinates.
            auto reflected_y = raw_y;
            reflect_diagram_vector(n, 1, reflected_y);
            transform(n, 1, reflected_y);
            auto got = canonical_bilinear(n, 1, x, 0, reflected_y, 0);

            if (want != got) {
                std::cerr << "midpoint mismatch n=" << n
                          << " trial=" << trial
                          << " want=" << want << " got=" << got << "\n";
                return 1;
            }

            auto roundtrip = x;
            inverse_transform(n, 1, roundtrip);
            transform(n, 1, roundtrip);
            if (roundtrip != x) {
                std::cerr << "T^-1/T mismatch n=" << n << "\n";
                return 2;
            }
        }

        // Exact support enumeration is intentionally limited to small/medium
        // n.  At n=15 this is 1430 columns and is still a useful regression.
        std::uint64_t nnz = support_nnz(n);
        std::cout << "n=" << n << " dim=" << dim
                  << " closure_nnz=" << nnz;
        if (n < int(sizeof(RNA)/sizeof(RNA[0]))) {
            std::cout << " A004148=" << RNA[n]
                      << (nnz == RNA[n] ? " OK" : " MISMATCH");
            if (nnz != RNA[n]) return 3;
        }
        std::cout << "\n";
    }
    return 0;
}
