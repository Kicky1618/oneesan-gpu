#include <array>
#include <cstdint>
#include <iostream>
#include <limits>

namespace {

constexpr int MAX_W = 28;
constexpr int TABLE_ENTRIES = 1199;
using Rank64 = std::uint64_t;

constexpr std::uint32_t EMBEDDED[TABLE_ENTRIES] = {
#include "../../cuda/gridfp/gridfp_reduced_production_runtime_sector_offset_values.inc"
};
constexpr int ROW_BASE[11] = {0, 24, 59, 107, 170, 250, 349, 469, 612, 780, 975};
static_assert(sizeof(EMBEDDED) == 4796);

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
    int sequential_index = 0;

    for (int wi = 0; wi < 11; ++wi) {
        const int W = 8 + 2 * wi;
        const int K = (W - 2) / 2;
        const int L = K + 2;
        const int O = W - L;
        if (ROW_BASE[wi] != sequential_index) return 2;
        for (int r = 0; r <= O; ++r) {
            for (int local = 0; local <= L; ++local) {
                const Rank64 off = sector_offset(p, L, r, local);
                if (off > std::numeric_limits<std::uint32_t>::max()) {
                    std::cerr << "sector offset does not fit uint32 W=" << W
                              << " outer=" << r << " local=" << local
                              << " offset=" << off << '\n';
                    return 3;
                }
                if (sequential_index >= TABLE_ENTRIES ||
                    EMBEDDED[sequential_index] != off) {
                    std::cerr << "embedded sector offset mismatch W=" << W
                              << " outer=" << r << " local=" << local
                              << " index=" << sequential_index
                              << " embedded=" << (sequential_index < TABLE_ENTRIES ? EMBEDDED[sequential_index] : 0)
                              << " expected=" << off << '\n';
                    return 4;
                }
                if (off > max_offset) max_offset = off;
                ++active_entries;
                ++sequential_index;
            }
        }
        ++configurations;
    }
    if (configurations != 11 || active_entries != TABLE_ENTRIES ||
        sequential_index != TABLE_ENTRIES || max_offset != 1805186805ULL)
        return 5;

    std::cout << "gridfp-runtime-sector-offset-table-proof OK"
              << " W_configs=" << configurations
              << " active_entries=" << active_entries
              << " table_entries=" << TABLE_ENTRIES
              << " table_bytes=" << sizeof(EMBEDDED)
              << " max_offset=" << max_offset
              << " row_bases_exact=1 uint32_exact=1 embedded_exact=1\n";
    return 0;
}
