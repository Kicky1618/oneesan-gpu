#include <array>
#include <cstdint>
#include <iostream>
#include <limits>

namespace {
using Rank64 = std::uint64_t;
constexpr int MAX_W = 28;
constexpr int SHIFT = 54;
constexpr std::array<Rank64, 11> TOTAL = {
    632ULL, 4451ULL, 32427ULL, 242413ULL, 1849269ULL, 14339193ULL,
    112685373ULL, 895517316ULL, 7184644894ULL, 58113695597ULL,
    473397057701ULL,
};
constexpr std::array<Rank64, 11> MAGIC54 = {
    28503795109939ULL, 4047269941469ULL, 555537006490ULL,
    74312840109ULL, 9741361862ULL, 1256304905ULL,
    159864568ULL, 20116192ULL, 2507347ULL, 309985ULL, 38053ULL,
};

Rank64 binom(int n, int k) {
    if (n < 0 || k < 0 || k > n) return 0;
    Rank64 x = 1;
    for (int i = 1; i <= k; ++i) x = x * Rank64(n - k + i) / Rank64(i);
    return x;
}

std::array<std::array<Rank64, MAX_W + 2>, MAX_W + 1> primitive_table() {
    std::array<std::array<Rank64, MAX_W + 2>, MAX_W + 1> p{};
    p[0][0] = 1;
    for (int rem = 1; rem <= MAX_W; ++rem) {
        for (int h = 0; h <= MAX_W; ++h) {
            p[rem][h] = p[rem - 1][h + 1] + (h ? p[rem - 1][h - 1] : 0);
        }
    }
    return p;
}

Rank64 group_size(
    const std::array<std::array<Rank64, MAX_W + 2>, MAX_W + 1>& p,
    int L,
    int r
) {
    Rank64 s = 0;
    for (int l = 0; l <= L; ++l) {
        const int occupied = r + l;
        if (!(occupied & 1)) continue;
        s += (binom(L, l) + binom(L - 2, l - 1)) * p[occupied][1];
    }
    return s;
}

Rank64 fixed_div54(Rank64 numerator, Rank64 magic) {
    return (numerator * magic) >> SHIFT;
}
}  // namespace

int main() {
    static_assert((__uint128_t(16) << SHIFT) < (__uint128_t(1) << 64));

    const auto primitive = primitive_table();
    std::uint64_t groups = 0;
    std::uint64_t owner_cases = 0;
    std::uint64_t shift53_failures = 0;
    Rank64 max_numerator = 0;
    Rank64 max_product = 0;

    for (int wi = 0; wi < 11; ++wi) {
        const int W = 8 + 2 * wi;
        const int L = W / 2 + 1;
        const int O = W - L;
        const Rank64 total = TOTAL[wi];
        const Rank64 expected_magic = (Rank64(1) << SHIFT) / total;
        if (expected_magic != MAGIC54[wi]) {
            std::cerr << "magic54 mismatch W=" << W << '\n';
            return 2;
        }

        Rank64 rebuilt_total = 0;
        for (int r = 0; r <= O; ++r) {
            rebuilt_total += binom(O, r) * group_size(primitive, L, r);
        }
        if (rebuilt_total != total) {
            std::cerr << "total mismatch W=" << W << " got=" << rebuilt_total
                      << " table=" << total << '\n';
            return 3;
        }

        Rank64 prefix = 0;
        for (int r = 0; r <= O; ++r) {
            const Rank64 group = group_size(primitive, L, r);
            const Rank64 count = binom(O, r);
            for (Rank64 support_rank = 0; support_rank < count; ++support_rank) {
                const Rank64 base = prefix + support_rank * group;
                const Rank64 midpoint = base + group / 2;
                ++groups;

                for (int ngpu = 2; ngpu <= 16; ++ngpu) {
                    const Rank64 numerator = midpoint * Rank64(ngpu);
                    const __uint128_t wide_product =
                        __uint128_t(numerator) * MAGIC54[wi];
                    if (wide_product > std::numeric_limits<Rank64>::max()) {
                        std::cerr << "64-bit product overflow W=" << W
                                  << " ngpu=" << ngpu << '\n';
                        return 4;
                    }

                    const Rank64 exact = numerator / total;
                    const Rank64 fast = fixed_div54(numerator, MAGIC54[wi]);
                    if (fast != exact || fast >= Rank64(ngpu)) {
                        std::cerr << "fixed54 owner mismatch W=" << W
                                  << " r=" << r
                                  << " support_rank=" << support_rank
                                  << " ngpu=" << ngpu
                                  << " numerator=" << numerator
                                  << " exact=" << exact
                                  << " fast=" << fast << '\n';
                        return 5;
                    }

                    const Rank64 magic53 = (Rank64(1) << 53) / total;
                    if (((numerator * magic53) >> 53) != exact) {
                        ++shift53_failures;
                    }

                    max_numerator = std::max(max_numerator, numerator);
                    max_product = std::max(max_product, Rank64(wide_product));
                    ++owner_cases;
                }
            }
            prefix += count * group;
        }
        if (prefix != total) return 6;
    }

    if (groups != 16376ULL || owner_cases != 245640ULL) return 7;
    if (shift53_failures == 0) return 8;

    std::cout << "gridfp-runtime-owner-fixed54-proof OK"
              << " W_min=8 W_max=28 W_step=2"
              << " groups=" << groups
              << " owner_cases=" << owner_cases
              << " shift=" << SHIFT
              << " shift53_failures=" << shift53_failures
              << " max_numerator=" << max_numerator
              << " max_product=" << max_product
              << " product_bits=58"
              << " table_entries=" << MAGIC54.size()
              << " table_bytes=" << MAGIC54.size() * sizeof(Rank64)
              << " mulhi=0 correction=0 exact=1\n";
    return 0;
}
