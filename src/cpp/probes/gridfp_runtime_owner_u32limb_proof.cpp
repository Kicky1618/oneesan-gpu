#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>

namespace {
using Rank64 = std::uint64_t;
constexpr int MAX_W = 28;
constexpr int MAX_GPU = 8;
constexpr std::array<Rank64, 11> TOTAL = {
    632ULL,4451ULL,32427ULL,242413ULL,1849269ULL,14339193ULL,
    112685373ULL,895517316ULL,7184644894ULL,58113695597ULL,
    473397057701ULL
};
// low16=magic, high16=minimum exact shift for the production midpoint set.
constexpr std::array<std::uint32_t, 11> META = {
    1246013u,1245301u,1573381u,1970509u,2032777u,2163287u,
    2631197u,2757423u,2954017u,3150571u,3417385u
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

Rank64 owner_u32limb(Rank64 numerator, std::uint32_t meta) {
    const unsigned shift = meta >> 16;
    const std::uint32_t magic = meta & 0xffffu;
    const std::uint32_t lo = static_cast<std::uint32_t>(numerator);
    if (shift < 32)
        return (Rank64(lo) * magic) >> shift;
    const std::uint32_t hi = static_cast<std::uint32_t>(numerator >> 32);
    const std::uint32_t upper =
        hi * magic + static_cast<std::uint32_t>((Rank64(lo) * magic) >> 32);
    return upper >> (shift - 32);
}
}  // namespace

int main() {
    const auto p = primitive_table();
    std::uint64_t groups = 0, owner_cases = 0, previous_shift_failures = 0;
    std::uint32_t max_magic = 0, max_upper = 0;
    unsigned max_shift = 0;

    for (int wi = 0; wi < 11; ++wi) {
        const int W = 8 + 2 * wi;
        const int L = W / 2 + 1;
        const int O = W - L;
        Rank64 total = 0;
        for (int r = 0; r <= O; ++r)
            total += binom(O, r) * group_size(p, L, r);
        if (total != TOTAL[wi]) return 2;

        const std::uint32_t meta = META[wi];
        const unsigned shift = meta >> 16;
        const std::uint32_t magic = meta & 0xffffu;
        if (magic != ((Rank64(1) << shift) / total)) return 3;
        max_magic = std::max(max_magic, magic);
        max_shift = std::max(max_shift, shift);

        Rank64 prefix = 0;
        for (int r = 0; r <= O; ++r) {
            const Rank64 group = group_size(p, L, r);
            const Rank64 count = binom(O, r);
            for (Rank64 sr = 0; sr < count; ++sr) {
                const Rank64 midpoint = prefix + sr * group + group / 2;
                ++groups;
                for (int ngpu = 2; ngpu <= MAX_GPU; ++ngpu) {
                    const Rank64 numerator = midpoint * Rank64(ngpu);
                    const Rank64 exact = numerator / total;
                    const Rank64 fast = owner_u32limb(numerator, meta);
                    if (fast != exact || fast >= Rank64(ngpu)) return 4;

                    const unsigned previous_shift = shift - 1;
                    const Rank64 previous_magic =
                        (Rank64(1) << previous_shift) / total;
                    if (((numerator * previous_magic) >> previous_shift) != exact)
                        ++previous_shift_failures;

                    if (shift >= 32) {
                        const std::uint32_t lo = static_cast<std::uint32_t>(numerator);
                        const std::uint32_t hi = static_cast<std::uint32_t>(numerator >> 32);
                        const std::uint32_t upper = hi * magic +
                            static_cast<std::uint32_t>((Rank64(lo) * magic) >> 32);
                        max_upper = std::max(max_upper, upper);
                    }
                    ++owner_cases;
                }
            }
            prefix += count * group;
        }
        if (prefix != total) return 5;
    }

    if (groups != 16376ULL || owner_cases != 114632ULL ||
        previous_shift_failures != 37ULL || max_magic != 9757u ||
        max_upper != 8372291u) return 6;

    std::cout << "gridfp-runtime-owner-u32limb-proof OK"
              << " W_min=8 W_max=28 W_step=2"
              << " ngpu_min=2 ngpu_max=" << MAX_GPU
              << " groups=" << groups
              << " owner_cases=" << owner_cases
              << " previous_shift_failures=" << previous_shift_failures
              << " max_magic=" << max_magic
              << " max_magic_bits=14"
              << " max_shift=" << max_shift
              << " max_upper=" << max_upper
              << " max_upper_bits=23"
              << " table_entries=" << META.size()
              << " table_bytes=" << META.size() * sizeof(std::uint32_t)
              << " mul64_general=0 clamp_required=0 exact=1\n";
    return 0;
}
