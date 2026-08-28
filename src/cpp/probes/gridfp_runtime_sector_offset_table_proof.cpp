#include <array>
#include <cstdint>
#include <iostream>
#include <limits>

namespace {

constexpr int MAX_W = 28;
constexpr int OUTER_SLOTS = 14;
constexpr int LOCAL_SLOTS = 16;
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

Rank64 sector_offset(
    const std::array<std::array<Rank64, MAX_W + 2>, MAX_W + 1>& p,
    int L, int outer, int local_ones
) {
    Rank64 off = 0;
    for (int l = 0; l < local_ones; ++l) {
        const int occupied = outer + l;
        if (!(occupied & 1)) continue;
        off += (binom(L, l) + binom(L - 2, l - 1)) * p[occupied][1];
    }
    return off;
}

} // namespace

int main() {
    const auto p = primitive_table();
    std::uint64_t configurations = 0;
    std::uint64_t active_entries = 0;
    Rank64 max_offset = 0;
    for (int W = 8; W <= MAX_W; W += 2) {
        const int K = (W - 2) / 2;
        const int L = K + 2;
        const int O = W - L;
        std::uint32_t table[OUTER_SLOTS][LOCAL_SLOTS]{};
        for (int r = 0; r <= O; ++r) {
            for (int local = 0; local <= L; ++local) {
                const Rank64 off = sector_offset(p, L, r, local);
                if (off > std::numeric_limits<std::uint32_t>::max()) {
                    std::cerr << "sector offset does not fit uint32 W=" << W
                              << " outer=" << r << " local=" << local
                              << " offset=" << off << '\n';
                    return 2;
                }
                table[r][local] = static_cast<std::uint32_t>(off);
                if (off > max_offset) max_offset = off;
                ++active_entries;
            }
        }
        for (int r = 0; r <= O; ++r)
            for (int local = 0; local <= L; ++local)
                if (table[r][local] != sector_offset(p, L, r, local)) return 3;
        ++configurations;
    }
    if (configurations != 11 || active_entries != 1199 ||
        max_offset != 1805186805ULL) return 4;

    std::cout << "gridfp-runtime-sector-offset-table-proof OK"
              << " W_configs=" << configurations
              << " active_entries=" << active_entries
              << " runtime_table_entries=" << OUTER_SLOTS * LOCAL_SLOTS
              << " runtime_table_bytes=" << OUTER_SLOTS * LOCAL_SLOTS * sizeof(std::uint32_t)
              << " max_offset=" << max_offset
              << " uint32_exact=1 table_exact=1\n";
    return 0;
}
