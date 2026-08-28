#include <array>
#include <cstdint>
#include <iostream>

namespace {
using Rank64 = std::uint64_t;
static constexpr int MAX = 28;
std::array<std::array<Rank64, MAX + 2>, MAX + 1> P{};

void build() {
    P[0][0] = 1;
    for (int rem = 1; rem <= MAX; ++rem) {
        for (int h = 0; h <= MAX; ++h) {
            Rank64 z = P[rem - 1][h + 1];
            if (h > 0) z += P[rem - 1][h - 1];
            P[rem][h] = z;
        }
    }
}
}

int main() {
    build();
    constexpr std::array<std::uint32_t, 14> expected{
        1u, 2u, 5u, 14u, 42u, 132u, 429u, 1430u,
        4862u, 16796u, 58786u, 208012u, 742900u, 2674440u};
    std::uint32_t max_value = 0;
    for (int sector = 0; sector < 14; ++sector) {
        const int occupied = 1 + 2 * sector;
        const Rank64 v = P[occupied][1];
        if (v != expected[std::size_t(sector)]) {
            std::cerr << "primitive1 mismatch sector=" << sector
                      << " occupied=" << occupied << " got=" << v
                      << " expected=" << expected[std::size_t(sector)] << '\n';
            return 2;
        }
        if (v > 0xffffffffULL) return 3;
        if (v > max_value) max_value = static_cast<std::uint32_t>(v);
    }
    if (max_value != 2674440u) return 4;
    std::cout << "gridfp-primitive1-u32-table-proof OK"
              << " sectors=14"
              << " occupied_min=1 occupied_max=27"
              << " table_entries=14 table_bytes=56"
              << " max_value=" << max_value
              << " uint32_exact=1 exact=1\n";
    return 0;
}
