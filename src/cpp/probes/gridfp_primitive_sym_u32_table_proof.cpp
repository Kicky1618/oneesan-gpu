#include <array>
#include <cstdint>
#include <iostream>

namespace {
using Rank64 = std::uint64_t;
static constexpr int MAX = 28;
static constexpr int COLS = MAX + 2;
static constexpr int COMPACT_ENTRIES = 225;
static constexpr int FULL_ENTRIES = (MAX + 1) * COLS;
static constexpr std::array<std::uint32_t, COMPACT_ENTRIES> COMPACT = {
#include "../../cuda/gridfp/gridfp_reduced_production_primitive_sym_u32_values.inc"
};
static constexpr std::array<std::uint32_t, FULL_ENTRIES> FULL = {
#include "../../cuda/gridfp/gridfp_reduced_production_primitive_full_u32_values_0_9.inc"
,
#include "../../cuda/gridfp/gridfp_reduced_production_primitive_full_u32_values_10_19.inc"
,
#include "../../cuda/gridfp/gridfp_reduced_production_primitive_full_u32_values_20_28.inc"
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
std::uint32_t full_primitive(int rem, int h) {
    if (rem < 0 || rem > MAX || h < 0 || h >= COLS) return 0;
    return FULL[std::size_t(rem * COLS + h)];
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
            const std::uint32_t compact = compact_primitive(rem, h);
            const std::uint32_t full = full_primitive(rem, h);
            if (ref != compact || ref != full) {
                std::cerr << "primitive mismatch rem=" << rem << " h=" << h
                          << " ref=" << ref << " compact=" << compact
                          << " full=" << full << '\n';
                return 3;
            }
            ++table_cases;
            if (ref) {
                ++nonzero_cases;
                if (ref > 0xffffffffULL) return 4;
                if (compact > max_value) max_value = compact;
            }
        }
    }
    if (expected_base != COMPACT_ENTRIES) return 5;
    if (table_cases != 870ULL) return 6;
    if (nonzero_cases != 225ULL) return 7;
    if (max_value != 8947575u) return 8;
    constexpr std::uint64_t old_bytes = 29ULL * 30ULL * 8ULL;
    constexpr std::uint64_t compact_bytes = COMPACT_ENTRIES * 4ULL;
    constexpr std::uint64_t full_bytes = FULL_ENTRIES * 4ULL;
    static_assert(old_bytes == 6960);
    static_assert(compact_bytes == 900);
    static_assert(full_bytes == 3480);
    std::cout << "gridfp-primitive-sym-u32-table-proof OK"
              << " production_rem_max=28"
              << " table_cases=" << table_cases
              << " nonzero_cells=" << nonzero_cases
              << " compact_entries=" << COMPACT_ENTRIES
              << " full_entries=" << FULL_ENTRIES
              << " old_bytes=" << old_bytes
              << " compact_bytes=" << compact_bytes
              << " full_bytes=" << full_bytes
              << " compact_saved_bytes=" << (old_bytes - compact_bytes)
              << " full_saved_bytes=" << (old_bytes - full_bytes)
              << " max_value=" << max_value
              << " parity_sparse_exact=1 full_shape_exact=1"
              << " row_base_exact=1 uint32_exact=1 exact=1\n";
    return 0;
}
