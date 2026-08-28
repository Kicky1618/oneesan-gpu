#include <array>
#include <cstdint>
#include <iostream>

namespace {
using Rank64 = std::uint64_t;
std::array<std::array<Rank64, 14>, 14> C{};

void build_choose() {
    C[0][0] = 1;
    for (int n = 1; n <= 13; ++n) {
        C[n][0] = C[n][n] = 1;
        for (int k = 1; k < n; ++k) C[n][k] = C[n - 1][k - 1] + C[n - 1][k];
    }
}

std::uint16_t unrank13(int ones, Rank64 rank) {
    std::uint16_t mask = 0;
    int left = ones;
    for (int pos = 0; pos < 13; ++pos) {
        if (!left) break;
        const int remaining = 13 - pos;
        if (left == remaining) {
            const std::uint16_t suffix = std::uint16_t(((1u << remaining) - 1u) << pos);
            mask = std::uint16_t(mask | suffix);
            break;
        }
        const int rem = remaining - 1;
        const Rank64 zero_count = C[rem][left];
        if (rank < zero_count) continue;
        rank -= zero_count;
        mask = std::uint16_t(mask | (std::uint16_t(1u) << pos));
        --left;
    }
    return mask;
}

Rank64 rank13(std::uint16_t mask, int ones) {
    Rank64 rank = 0;
    int left = ones;
    for (int pos = 0; pos < 13; ++pos) {
        if (((mask >> pos) & 1u) == 0) continue;
        rank += C[12 - pos][left];
        --left;
    }
    return rank;
}

} // namespace

int main() {
    build_choose();
    constexpr std::array<int, 14> base{
        0, 1, 14, 92, 378, 1093, 2380,
        4096, 5812, 7099, 7814, 8100, 8178, 8191};
    std::array<std::uint16_t, 8192> table{};
    std::array<bool, 8192> mask_seen{};
    std::uint64_t checked = 0;

    for (int ones = 0; ones <= 13; ++ones) {
        const Rank64 count = C[13][ones];
        if (base[std::size_t(ones)] + int(count) > 8192) return 2;
        for (Rank64 rank = 0; rank < count; ++rank) {
            const std::uint16_t mask = unrank13(ones, rank);
            if (__builtin_popcount(unsigned(mask)) != ones) return 3;
            if (rank13(mask, ones) != rank) {
                std::cerr << "rank roundtrip mismatch ones=" << ones
                          << " rank=" << rank << " mask=" << mask << '\n';
                return 4;
            }
            const int ix = base[std::size_t(ones)] + int(rank);
            table[std::size_t(ix)] = mask;
            if (mask_seen[std::size_t(mask)]) {
                std::cerr << "duplicate mask=" << mask << '\n';
                return 5;
            }
            mask_seen[std::size_t(mask)] = true;
            ++checked;
        }
    }

    if (checked != 8192) return 6;
    for (bool x : mask_seen) if (!x) return 7;
    if (table.front() != 0u || table.back() != 8191u) return 8;

    std::cout << "gridfp-support-unrank-len13-table-proof OK"
              << " ranks=8192 unique_masks=8192 bits=13"
              << " table_entry_bits=16 table_bytes=" << sizeof(table)
              << " max_linear_iterations=13 rank_roundtrip_exact=1"
              << " coverage_all_masks=1\n";
    return 0;
}
