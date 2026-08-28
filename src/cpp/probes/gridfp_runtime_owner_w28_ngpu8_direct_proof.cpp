#include <array>
#include <cstdint>
#include <iostream>

namespace {
using Rank64 = std::uint64_t;
constexpr int W = 28;
constexpr int L = 15;
constexpr int O = 13;
constexpr int NGPU = 8;
constexpr Rank64 TOTAL = 473397057701ULL;
constexpr std::uint32_t MAGIC = 9513u;
constexpr unsigned SHIFT = 17;
constexpr int MAX_W = 28;

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

std::uint32_t owner_direct(Rank64 midpoint) {
    const std::uint32_t lo = static_cast<std::uint32_t>(midpoint);
    const std::uint32_t hi = static_cast<std::uint32_t>(midpoint >> 32);
    const std::uint32_t product_hi =
        static_cast<std::uint32_t>((Rank64(lo) * MAGIC) >> 32);
    return (hi * MAGIC + product_hi) >> SHIFT;
}
}  // namespace

int main() {
    const auto p = primitive_table();
    Rank64 total = 0;
    for (int r = 0; r <= O; ++r)
        total += binom(O, r) * group_size(p, r);
    if (total != TOTAL) return 2;

    Rank64 prefix = 0;
    std::uint64_t groups = 0;
    std::uint32_t max_owner = 0;
    for (int r = 0; r <= O; ++r) {
        const Rank64 group = group_size(p, r);
        const Rank64 count = binom(O, r);
        for (Rank64 sr = 0; sr < count; ++sr) {
            const Rank64 midpoint = prefix + sr * group + group / 2;
            const Rank64 exact = (midpoint * Rank64(NGPU)) / TOTAL;
            const std::uint32_t fast = owner_direct(midpoint);
            if (fast != exact || fast >= NGPU) return 3;
            if (fast > max_owner) max_owner = fast;
            ++groups;
        }
        prefix += count * group;
    }
    if (prefix != TOTAL || groups != 8192ULL || max_owner != 7u) return 4;

    std::cout << "gridfp-runtime-owner-w28-ngpu8-direct-proof OK"
              << " W=28 ngpu=8 groups=" << groups
              << " total=" << TOTAL
              << " magic=" << MAGIC
              << " shift=" << SHIFT
              << " max_owner=" << max_owner
              << " meta_loads=0 variable_shift=0 ngpu_mul=0 scale_mul=0"
              << " product_lo=0 mul64=0 clamp=0 exact=1\n";
    return 0;
}
