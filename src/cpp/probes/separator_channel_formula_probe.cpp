#include <array>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

using U64 = std::uint64_t;

// Conjectured universal separator-channel rank before finite-side saturation:
//   C(r,h) = [x^(r-h)] (1 - 2x - 3x^2)^(-(h+1)/2).
// These are exactly OEIS A111960, the renewal/convolution triangle of the
// central trinomial coefficients.
static U64 channel(int r, int h) {
    if (h < 0 || h > r) return 0;
    int nmax = r - h;
    std::vector<U64> c(nmax + 1, 0);
    c[0] = 1;
    // If c_n=[x^n](1-2x-3x^2)^(-(h+1)/2), differentiation gives
    // (n+1)c_{n+1}=(2n+h+1)c_n+3(n+h)c_{n-1}.
    for (int n = 0; n < nmax; ++n) {
        __uint128_t z = __uint128_t(2 * n + h + 1) * c[n];
        if (n) z += __uint128_t(3 * (n + h)) * c[n - 1];
        if (z % U64(n + 1)) {
            std::cerr << "nonintegral recurrence r=" << r << " h=" << h
                      << " n=" << n << "\n";
            std::exit(2);
        }
        c[n + 1] = U64(z / U64(n + 1));
    }
    return c[nmax];
}

int main() {
    std::cout << "separator renewal triangle C(r,h)\n";
    for (int r = 0; r <= 14; ++r) {
        U64 sum = 0;
        std::cout << "r=" << std::setw(2) << r << " :";
        for (int h = 0; h <= r; ++h) {
            U64 z = channel(r, h);
            sum += z;
            std::cout << " " << z;
        }
        std::cout << "  sum=" << sum << "\n";
    }

    // Balanced 14|14 segment dimensions for the W=28 one-defect Motzkin space.
    // H[h] and L[h] are the two side dimensions in the conventions used by the
    // Schmidt-rank experiments.  Finite-side saturation predicts
    // rank(r,h)=min(C(r,h),H[h],L[h]).
    constexpr std::array<U64, 15> H = {
        196938,345957,417522,409500,343278,250887,161070,90909,
        44928,19278,7084,2183,546,105,14
    };
    constexpr std::array<U64, 15> L = {
        113634,196938,232323,220584,177177,122694,73710,38376,
        17199,6552,2079,532,104,14,1
    };
    constexpr U64 FULL = 385719506620ULL;

    std::cout << "\nW=28 balanced-cut projection\n";
    for (int r = 0; r <= 14; ++r) {
        U64 rank_sum = 0;
        __uint128_t factor = 0;
        for (int h = 0; h <= 14; ++h) {
            U64 q = h <= r ? channel(r, h) : 0;
            U64 rk = std::min(q, std::min(H[h], L[h]));
            rank_sum += rk;
            factor += __uint128_t(rk) * (H[h] + L[h]);
        }
        U64 fe = U64(factor);
        double ratio = double(fe) / double(FULL);
        double gib = double(fe) * 4.0 / double(1ULL << 30);
        std::cout << "row=" << std::setw(2) << r
                  << " rank_sum=" << std::setw(8) << rank_sum
                  << " factor_entries=" << std::setw(12) << fe
                  << " factor/full=" << std::fixed << std::setprecision(6)
                  << ratio
                  << " GiB/residue=" << std::setprecision(3) << gib << "\n";
    }
}
