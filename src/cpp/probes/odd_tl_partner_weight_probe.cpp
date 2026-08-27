#include <cassert>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#define main odd_tl_integer_probe_unused_main
#include "odd_tl_integer_normalization_probe.cpp"
#undef main

struct WeightedInvolution {
    std::vector<std::uint32_t> partner;
    std::vector<std::uint32_t> weight;
};

static WeightedInvolution build_weighted_involution(int n, int d) {
    int dimv = int(basis(n, d).words.size());
    WeightedInvolution out;
    out.partner.assign(dimv, 0xffffffffu);
    out.weight.assign(dimv, 0);

    if (n == 0) {
        if (d == 0) {
            out.partner[0] = 0;
            out.weight[0] = 1;
        }
        return out;
    }
    if (n == 1) {
        if (d == 1) {
            out.partner[0] = 0;
            out.weight[0] = 1;
        }
        return out;
    }

    int da = int(basis(n - 2, d - 2).words.size());
    int d0 = int(basis(n - 2, d).words.size());
    int dp = int(basis(n - 2, d + 2).words.size());
    int offA = 0;
    int offB = da;
    int offC = da + d0;
    int offD = da + 2 * d0;
    std::uint32_t s = std::uint32_t((d + 1) / 2);
    std::uint32_t sinv = invm(s);

    if (da) {
        auto child = build_weighted_involution(n - 2, d - 2);
        std::uint32_t scale = mulm(sinv, sinv);
        for (int i = 0; i < da; ++i) {
            out.partner[offA + i] = offA + child.partner[i];
            out.weight[offA + i] = mulm(scale, child.weight[i]);
        }
    }

    if (d0) {
        auto child = build_weighted_involution(n - 2, d);
        for (int i = 0; i < d0; ++i) {
            // H tensor D_child: B and C are exchanged, while the child partner
            // is applied inside each copy.
            out.partner[offB + i] = offC + child.partner[i];
            out.partner[offC + i] = offB + child.partner[i];
            out.weight[offB + i] = child.weight[i];
            out.weight[offC + i] = child.weight[i];
        }
    }

    if (dp) {
        auto child = build_weighted_involution(n - 2, d + 2);
        std::uint32_t scale = negm(mulm(std::uint32_t((d + 3) / 2), sinv));
        for (int i = 0; i < dp; ++i) {
            out.partner[offD + i] = offD + child.partner[i];
            out.weight[offD + i] = mulm(scale, child.weight[i]);
        }
    }
    return out;
}

static std::uint32_t flat_bilinear(
    WeightedInvolution const& g,
    std::vector<std::uint32_t> const& x,
    std::vector<std::uint32_t> const& y
) {
    assert(g.partner.size() == x.size() && x.size() == y.size());
    std::uint32_t z = 0;
    for (std::uint32_t i = 0; i < g.partner.size(); ++i) {
        auto p = g.partner[i];
        z = addm(z, mulm(g.weight[i], mulm(x[i], y[p])));
    }
    return z;
}

int main(int argc, char** argv) {
    int maxn = argc > 1 ? std::atoi(argv[1]) : 15;
    if ((maxn & 1) == 0) --maxn;
    std::mt19937_64 rng(0x3141592653589793ULL);

    std::uint64_t total_entries = 0;
    for (int n = 1; n <= maxn; n += 2) {
        for (int d = 1; d <= n; d += 2) {
            int dimv = int(basis(n, d).words.size());
            if (!dimv) continue;
            auto g = build_weighted_involution(n, d);
            total_entries += g.partner.size();

            for (int i = 0; i < dimv; ++i) {
                auto p = g.partner[i];
                if (p >= g.partner.size() || g.partner[p] != std::uint32_t(i)) {
                    std::cerr << "partner not involutive n=" << n << " d=" << d
                              << " i=" << i << " p=" << p << "\n";
                    return 1;
                }
                if (g.weight[i] != g.weight[p]) {
                    std::cerr << "asymmetric weight n=" << n << " d=" << d
                              << " i=" << i << " p=" << p << "\n";
                    return 2;
                }
            }

            for (int trial = 0; trial < 4; ++trial) {
                std::vector<std::uint32_t> x(dimv), y(dimv);
                for (auto& q : x) q = std::uint32_t(rng() % MOD);
                for (auto& q : y) q = std::uint32_t(rng() % MOD);
                auto want = canonical_bilinear_integer(n, d, x, 0, y, 0);
                auto got = flat_bilinear(g, x, y);
                if (want != got) {
                    std::cerr << "flat Gram mismatch n=" << n << " d=" << d
                              << " trial=" << trial << " want=" << want
                              << " got=" << got << "\n";
                    return 3;
                }
            }
            std::cout << "OK weighted-involution n=" << n << " d=" << d
                      << " dim=" << dimv << "\n";
        }
    }
    std::cout << "PASS weighted involution total_entries=" << total_entries << "\n";
    return 0;
}
