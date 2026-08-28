#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>

namespace {
using Rank64 = std::uint64_t;
constexpr int MAX_W = 28;
constexpr int MAX_GPU = 8;
constexpr int SHIFT = 52;
constexpr std::array<Rank64, 11> TOTAL = {
    632ULL,4451ULL,32427ULL,242413ULL,1849269ULL,14339193ULL,
    112685373ULL,895517316ULL,7184644894ULL,58113695597ULL,
    473397057701ULL
};
constexpr std::array<Rank64, 11> MAGIC52 = {
    7125948777484ULL,1011817485367ULL,138884251622ULL,
    18578210027ULL,2435340465ULL,314076226ULL,
    39966142ULL,5029048ULL,626836ULL,77496ULL,9513ULL
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
    for (int rem = 1; rem <= MAX_W; ++rem)
        for (int h = 0; h <= MAX_W; ++h)
            p[rem][h] = p[rem - 1][h + 1] + (h ? p[rem - 1][h - 1] : 0);
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
}  // namespace

int main() {
    const auto p = primitive_table();
    std::uint64_t groups = 0;
    std::uint64_t owner_cases = 0;
    std::uint64_t shift51_failures = 0;
    Rank64 max_numerator = 0;
    Rank64 max_product = 0;

    for (int wi = 0; wi < 11; ++wi) {
        const int W = 8 + 2 * wi;
        const int L = W / 2 + 1;
        const int O = W - L;
        Rank64 total = 0;
        for (int r = 0; r <= O; ++r)
            total += binom(O, r) * group_size(p, L, r);
        if (total != TOTAL[wi]) return 2;

        const Rank64 expected_magic = (Rank64(1) << SHIFT) / total;
        if (expected_magic != MAGIC52[wi]) return 3;

        Rank64 prefix = 0;
        for (int r = 0; r <= O; ++r) {
            const Rank64 group = group_size(p, L, r);
            const Rank64 count = binom(O, r);
            for (Rank64 sr = 0; sr < count; ++sr) {
                const Rank64 midpoint = prefix + sr * group + group / 2;
                ++groups;
                for (int ngpu = 2; ngpu <= MAX_GPU; ++ngpu) {
                    const Rank64 numerator = midpoint * Rank64(ngpu);
                    const __uint128_t wide = __uint128_t(numerator) * MAGIC52[wi];
                    if (wide > std::numeric_limits<Rank64>::max()) return 4;

                    const Rank64 exact = numerator / total;
                    const Rank64 fast = (numerator * MAGIC52[wi]) >> SHIFT;
                    if (fast != exact || fast >= Rank64(ngpu)) return 5;

                    const Rank64 magic51 = (Rank64(1) << 51) / total;
                    if (((numerator * magic51) >> 51) != exact)
                        ++shift51_failures;
                    max_numerator = std::max(max_numerator, numerator);
                    max_product = std::max(max_product, Rank64(wide));
                    ++owner_cases;
                }
            }
            prefix += count * group;
        }
        if (prefix != total) return 6;
    }

    if (groups != 16376ULL || owner_cases != 114632ULL ||
        shift51_failures != 3ULL) return 7;

    std::cout << "gridfp-runtime-owner-fixed52-proof OK"
              << " W_min=8 W_max=28 W_step=2"
              << " ngpu_min=2 ngpu_max=" << MAX_GPU
              << " groups=" << groups
              << " owner_cases=" << owner_cases
              << " shift=" << SHIFT
              << " shift51_failures=" << shift51_failures
              << " max_numerator=" << max_numerator
              << " max_product=" << max_product
              << " product_bits=55"
              << " table_entries=" << MAGIC52.size()
              << " table_bytes=" << MAGIC52.size() * sizeof(Rank64)
              << " owner_lt_ngpu=1 clamp_required=0"
              << " mulhi=0 correction=0 exact=1\n";
    return 0;
}
