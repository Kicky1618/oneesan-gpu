#include <array>
#include <cstdint>
#include <iostream>

namespace {
using Rank64 = std::uint64_t;
static constexpr int MAX = 28;
static constexpr int COLS = MAX + 2;
static constexpr int ENTRIES = 225;
static constexpr std::array<std::uint32_t, ENTRIES> COMPACT = {
#include "../../cuda/gridfp/gridfp_reduced_production_primitive_sym_u32_values.inc"
};

std::array<std::array<Rank64, COLS>, MAX + 1> build_primitive() {
    std::array<std::array<Rank64, COLS>, MAX + 1> p{};
    p[0][0] = 1;
    for (int rem = 1; rem <= MAX; ++rem) {
        for (int h = 0; h <= MAX; ++h) {
            Rank64 z = p[rem - 1][h + 1];
            if (h > 0) z += p[rem - 1][h - 1];
            p[rem][h] = z;
        }
    }
    return p;
}

constexpr int row_base(int rem) {
    const int m = rem >> 1;
    return (rem & 1) ? (m + 1) * (m + 1) : m * (m + 1);
}

std::uint32_t compact_primitive(int rem, int h) {
    if (rem < 0 || rem > MAX || h < 0 || h > rem || ((rem ^ h) & 1)) return 0;
    return COMPACT[std::size_t(row_base(rem) + (h >> 1))];
}
}

int main() {
    const auto primitive = build_primitive();
    std::uint64_t table_cases = 0;
    std::uint64_t nonzero_cases = 0;
    std::uint32_t max_value = 0;
    int expected_base = 0;
    for (int rem = 0; rem <= MAX; ++rem) {
        const int base = row_base(rem);
        if (base != expected_base) {
            std::cerr << "row base mismatch rem=" << rem << " got=" << base
                      << " expected=" << expected_base << '\n';
            return 2;
        }
        expected_base += rem / 2 + 1;
        for (int h = 0; h < COLS; ++h) {
            const Rank64 ref = primitive[rem][h];
            const std::uint32_t got = compact_primitive(rem, h);
            if (ref != got) {
                std::cerr << "primitive mismatch rem=" << rem << " h=" << h
                          << " ref=" << ref << " got=" << got << '\n';
                return 3;
            }
            ++table_cases;
            if (ref) {
                ++nonzero_cases;
                if (ref > 0xffffffffULL) return 4;
                if (got > max_value) max_value = got;
            }
        }
    }
    if (expected_base != ENTRIES) return 5;
    if (table_cases != 870ULL) return 6;
    if (nonzero_cases != 225ULL) return 7;
    if (max_value != 8947575u) return 8;
    constexpr std::uint64_t old_bytes = 29ULL * 30ULL * 8ULL;
    constexpr std::uint64_t new_bytes = ENTRIES * 4ULL;
    static_assert(old_bytes == 6960);
    static_assert(new_bytes == 900);
    std::cout << "gridfp-primitive-sym-u32-table-proof OK"
              << " production_rem_max=28"
              << " table_cases=" << table_cases
              << " nonzero_cells=" << nonzero_cases
              << " compact_entries=" << ENTRIES
              << " old_bytes=" << old_bytes
              << " compact_bytes=" << new_bytes
              << " saved_bytes=" << (old_bytes - new_bytes)
              << " max_value=" << max_value
              << " parity_sparse_exact=1 row_base_exact=1 uint32_exact=1 exact=1\n";
    return 0;
}
