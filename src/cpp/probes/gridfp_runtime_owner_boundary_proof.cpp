#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

namespace {

constexpr int MAX_W = 28;
using Rank64 = std::uint64_t;

Rank64 binom(int n, int k) {
    if (n < 0 || k < 0 || k > n) return 0;
    Rank64 x = 1;
    for (int i = 1; i <= k; ++i)
        x = x * Rank64(n - k + i) / Rank64(i);
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
    int L, int r
) {
    Rank64 total = 0;
    for (int l = 0; l <= L; ++l) {
        const int occupied = r + l;
        if (!(occupied & 1)) continue;
        total += (binom(L, l) + binom(L - 2, l - 1)) * p[occupied][1];
    }
    return total;
}

int boundary_owner(Rank64 group_base, const std::vector<Rank64>& owner_begin) {
    int owner = 0;
    for (int g = 1; g < static_cast<int>(owner_begin.size()); ++g) {
        const Rank64 b = owner_begin[static_cast<std::size_t>(g)];
        if (!b) continue;
        if (b > group_base) break;
        owner = g;
    }
    return owner;
}

} // namespace

int main() {
    const auto p = primitive_table();
    std::uint64_t configs = 0;
    std::uint64_t group_cases = 0;
    std::uint64_t empty_owner_slots = 0;
    std::uint64_t active_nonzero_owner_at_zero = 0;

    for (int W = 8; W <= MAX_W; W += 2) {
        const int K = (W - 2) / 2;
        const int L = K + 2;
        const int O = W - L;
        Rank64 total = 0;
        for (int r = 0; r <= O; ++r)
            total += binom(O, r) * group_size(p, L, r);

        for (int ngpu = 2; ngpu <= 16; ++ngpu) {
            std::vector<Rank64> begin(static_cast<std::size_t>(ngpu),
                                      std::numeric_limits<Rank64>::max());
            struct G { Rank64 base, size; int owner; };
            std::vector<G> groups;
            Rank64 prefix = 0;
            for (int r = 0; r <= O; ++r) {
                const Rank64 size = group_size(p, L, r);
                const Rank64 count = binom(O, r);
                for (Rank64 sr = 0; sr < count; ++sr) {
                    const Rank64 base = prefix + sr * size;
                    const Rank64 midpoint = base + size / 2;
                    int owner = int((static_cast<__uint128_t>(midpoint) * ngpu) / total);
                    if (owner >= ngpu) owner = ngpu - 1;
                    groups.push_back(G{base, size, owner});
                    auto& b = begin[static_cast<std::size_t>(owner)];
                    if (b == std::numeric_limits<Rank64>::max()) b = base;
                }
                prefix += count * size;
            }
            if (prefix != total) return 2;
            for (int g = 0; g < ngpu; ++g) {
                auto& b = begin[static_cast<std::size_t>(g)];
                if (b == std::numeric_limits<Rank64>::max()) {
                    b = 0;
                    ++empty_owner_slots;
                } else if (g > 0 && b == 0) {
                    ++active_nonzero_owner_at_zero;
                }
            }
            for (const auto& x : groups) {
                const int got = boundary_owner(x.base, begin);
                if (got != x.owner) {
                    std::cerr << "owner boundary mismatch W=" << W
                              << " ngpu=" << ngpu << " base=" << x.base
                              << " expected=" << x.owner << " got=" << got << '\n';
                    return 3;
                }
                ++group_cases;
            }
            ++configs;
        }
    }

    if (configs != 165 || active_nonzero_owner_at_zero != 0) return 4;
    std::cout << "gridfp-runtime-owner-boundary-proof OK"
              << " configs=" << configs
              << " group_cases=" << group_cases
              << " empty_owner_slots=" << empty_owner_slots
              << " active_nonzero_owner_at_zero=" << active_nonzero_owner_at_zero
              << " W_min=8 W_max=28 ngpu_min=2 ngpu_max=16"
              << " exact=1\n";
    return 0;
}
