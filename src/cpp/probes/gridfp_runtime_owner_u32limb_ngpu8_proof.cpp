#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>

namespace {
using Rank64 = std::uint64_t;
constexpr int MAX_W = 28;
constexpr int NGPU = 8;
constexpr std::array<Rank64, 11> TOTAL = {
    632ULL,4451ULL,32427ULL,242413ULL,1849269ULL,14339193ULL,
    112685373ULL,895517316ULL,7184644894ULL,58113695597ULL,
    473397057701ULL
};
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

std::uint32_t owner_ngpu8(Rank64 midpoint, std::uint32_t meta) {
    // (midpoint * magic * 8) >> shift == (midpoint * magic) >> (shift - 3).
    const unsigned shift = (meta >> 16) - 3;
    const std::uint32_t magic = meta & 0xffffu;
    const std::uint32_t lo = static_cast<std::uint32_t>(midpoint);
    const std::uint32_t product_lo = lo * magic;
    const std::uint32_t product_hi =
        static_cast<std::uint32_t>((Rank64(lo) * magic) >> 32);
    if (shift < 32)
        return (product_lo >> shift) | (product_hi << (32 - shift));
    const std::uint32_t hi = static_cast<std::uint32_t>(midpoint >> 32);
    const std::uint32_t upper = hi * magic + product_hi;
    return upper >> (shift - 32);
}
}  // namespace

int main() {
    const auto p = primitive_table();
    std::uint64_t groups = 0;
    unsigned min_effective_shift = 64, max_effective_shift = 0;
    std::uint32_t max_magic = 0, max_midpoint_hi = 0;

    for (int wi = 0; wi < 11; ++wi) {
        const int W = 8 + 2 * wi;
        const int L = W / 2 + 1;
        const int O = W - L;
        Rank64 total = 0;
        for (int r = 0; r <= O; ++r)
            total += binom(O, r) * group_size(p, L, r);
        if (total != TOTAL[wi]) return 2;

        const std::uint32_t meta = META[wi];
        const unsigned base_shift = meta >> 16;
        const unsigned shift = base_shift - 3;
        const std::uint32_t magic = meta & 0xffffu;
        if (magic != ((Rank64(1) << base_shift) / total)) return 3;
        min_effective_shift = std::min(min_effective_shift, shift);
        max_effective_shift = std::max(max_effective_shift, shift);
        max_magic = std::max(max_magic, magic);

        Rank64 prefix = 0;
        for (int r = 0; r <= O; ++r) {
            const Rank64 group = group_size(p, L, r);
            const Rank64 count = binom(O, r);
            for (Rank64 sr = 0; sr < count; ++sr) {
                const Rank64 midpoint = prefix + sr * group + group / 2;
                const Rank64 exact = (midpoint * Rank64(NGPU)) / total;
                const Rank64 fast = owner_ngpu8(midpoint, meta);
                if (fast != exact || fast >= Rank64(NGPU)) return 4;
                max_midpoint_hi = std::max(
                    max_midpoint_hi, static_cast<std::uint32_t>(midpoint >> 32));
                ++groups;
            }
            prefix += count * group;
        }
        if (prefix != total) return 5;
    }

    if (groups != 16376ULL || min_effective_shift != 16 ||
        max_effective_shift != 49 || max_magic != 9757u ||
        max_midpoint_hi != 110u) return 6;

    std::cout << "gridfp-runtime-owner-u32limb-ngpu8-proof OK"
              << " W_min=8 W_max=28 W_step=2 ngpu=8"
              << " groups=" << groups
              << " shift_bias=3"
              << " min_effective_shift=" << min_effective_shift
              << " max_effective_shift=" << max_effective_shift
              << " max_magic=" << max_magic
              << " max_magic_bits=14"
              << " max_midpoint_hi=" << max_midpoint_hi
              << " ngpu_mul=0 scale_mul=0 midpoint_mul64=0"
              << " pure_u32=1 exact=1\n";
    return 0;
}
