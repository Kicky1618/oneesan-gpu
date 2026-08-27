#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#define main odd_tl_rational_probe_unused_main
#include "odd_tl_gram_factorization_probe.cpp"
#undef main

// Q_d without its common denominator s=(d+1)/2.  All coefficients are small
// signed integers.  The corresponding coordinate is Ahat=s*A', and the Gram
// weight of the A child acquires 1/s^2.
static void apply_Q_integer(
    int n, int d,
    std::vector<std::uint32_t> const& src,
    std::vector<std::uint32_t>& dst
) {
    int din = d + 2;
    int r = (d - 1) / 2;
    if (r == 0) return;
    auto const& B = basis(n, din);

    for (int idx = 0; idx < int(B.words.size()); ++idx) {
        auto x = src[idx];
        if (!x) continue;

        for (int q = 0; q < 2 * r; ++q) {
            int to = map_two_cups(B.words[idx], n, din,
                                  {q, q + 3}, {q + 1, q + 2});
            assert(to >= 0);
            int num = q / 2 + 1;
            std::uint32_t c = std::uint32_t(num);
            if (q & 1) c = negm(c);
            add_scaled(dst, to, c, x);
        }

        for (int i = 0; i <= r; ++i) {
            for (int j = i + 1; j <= r; ++j) {
                int to = map_two_cups(B.words[idx], n, din,
                                      {2 * i, 2 * i + 1}, {2 * j + 1, 2 * j + 2});
                assert(to >= 0);
                int num = 1;
                int sign_exp = j - i;
                if (j == r) {
                    num = r;
                    sign_exp = r - 1 - i;
                }
                std::uint32_t c = std::uint32_t(num);
                if (sign_exp & 1) c = negm(c);
                add_scaled(dst, to, c, x);
            }
        }
    }
}

static void transform_integer(int n, int d, std::vector<std::uint32_t>& v) {
    if (n <= 1) return;
    int da = int(basis(n - 2, d - 2).words.size());
    int d0 = int(basis(n - 2, d).words.size());
    int dp = int(basis(n - 2, d + 2).words.size());
    assert(int(v.size()) == da + 2 * d0 + dp);

    std::vector<std::uint32_t> A(v.begin(), v.begin() + da);
    std::vector<std::uint32_t> B(v.begin() + da, v.begin() + da + d0);
    std::vector<std::uint32_t> C(v.begin() + da + d0, v.begin() + da + 2 * d0);
    std::vector<std::uint32_t> D(v.begin() + da + 2 * d0, v.end());

    const std::uint32_t s = std::uint32_t((d + 1) / 2);
    if (da) {
        // Ahat = s*A + s*partial(C) + Qtilde(D).
        for (auto& x : A) x = mulm(s, x);

        std::vector<std::uint32_t> t(da);
        apply_partial(n - 2, d, C, t);
        for (int i = 0; i < da; ++i) A[i] = addm(A[i], mulm(s, t[i]));

        std::fill(t.begin(), t.end(), 0);
        apply_Q_integer(n - 2, d, D, t);
        for (int i = 0; i < da; ++i) A[i] = addm(A[i], t[i]);
    }

    if (d0 && dp) {
        std::vector<std::uint32_t> t(d0);
        apply_partial(n - 2, d + 2, D, t);
        for (int i = 0; i < d0; ++i) B[i] = addm(B[i], t[i]);
        std::fill(t.begin(), t.end(), 0);
        apply_J(n - 2, d + 2, D, t);
        for (int i = 0; i < d0; ++i) C[i] = addm(C[i], t[i]);
    }

    if (da) transform_integer(n - 2, d - 2, A);
    if (d0) {
        transform_integer(n - 2, d, B);
        transform_integer(n - 2, d, C);
    }
    if (dp) transform_integer(n - 2, d + 2, D);

    int off = 0;
    for (auto x : A) v[off++] = x;
    for (auto x : B) v[off++] = x;
    for (auto x : C) v[off++] = x;
    for (auto x : D) v[off++] = x;
}

static std::uint32_t canonical_bilinear_integer(
    int n, int d,
    std::vector<std::uint32_t> const& x, int xo,
    std::vector<std::uint32_t> const& y, int yo
) {
    if (n == 0) return d == 0 ? mulm(x[xo], y[yo]) : 0;
    if (n == 1) return d == 1 ? mulm(x[xo], y[yo]) : 0;

    int da = int(basis(n - 2, d - 2).words.size());
    int d0 = int(basis(n - 2, d).words.size());
    int dp = int(basis(n - 2, d + 2).words.size());
    std::uint32_t z = 0;
    std::uint32_t s = std::uint32_t((d + 1) / 2);
    std::uint32_t sinv = invm(s);

    if (da) {
        auto q = canonical_bilinear_integer(n - 2, d - 2, x, xo, y, yo);
        z = addm(z, mulm(mulm(sinv, sinv), q));
    }
    if (d0) {
        z = addm(z, canonical_bilinear_integer(n - 2, d,
                                               x, xo + da,
                                               y, yo + da + d0));
        z = addm(z, canonical_bilinear_integer(n - 2, d,
                                               x, xo + da + d0,
                                               y, yo + da));
    }
    if (dp) {
        auto q = canonical_bilinear_integer(n - 2, d + 2,
                                            x, xo + da + 2 * d0,
                                            y, yo + da + 2 * d0);
        std::uint32_t c = negm(mulm(std::uint32_t((d + 3) / 2), sinv));
        z = addm(z, mulm(c, q));
    }
    return z;
}

int main() {
    std::mt19937_64 rng(0x1a2b3c4dULL);
    for (int n = 1; n <= 13; n += 2) {
        for (int d = 1; d <= n; d += 2) {
            int dimv = int(basis(n, d).words.size());
            if (!dimv) continue;
            for (int trial = 0; trial < 4; ++trial) {
                std::vector<std::uint32_t> x(dimv), y(dimv);
                for (auto& q : x) q = std::uint32_t(rng() % MOD);
                for (auto& q : y) q = std::uint32_t(rng() % MOD);

                auto want = explicit_bilinear(n, d, x, y);
                transform_integer(n, d, x);
                transform_integer(n, d, y);
                auto got = canonical_bilinear_integer(n, d, x, 0, y, 0);
                if (want != got) {
                    std::cerr << "MISMATCH n=" << n << " d=" << d
                              << " dim=" << dimv << " trial=" << trial
                              << " want=" << want << " got=" << got << "\n";
                    return 1;
                }
            }
            std::cout << "OK integer-normalized n=" << n
                      << " d=" << d << " dim=" << dimv << "\n";
        }
    }
    return 0;
}
