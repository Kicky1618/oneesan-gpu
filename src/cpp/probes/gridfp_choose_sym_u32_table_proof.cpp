#include <array>
#include <cstdint>
#include <iostream>

namespace {
using Rank64 = std::uint64_t;
static constexpr int MAX = 28;
static constexpr int ENTRIES = 225;
static constexpr std::array<std::uint32_t, ENTRIES> COMPACT = {
#include "../../cuda/gridfp/gridfp_reduced_production_choose_sym_u32_values.inc"
};

std::array<std::array<Rank64, MAX + 1>, MAX + 1> build_choose() {
    std::array<std::array<Rank64, MAX + 1>, MAX + 1> c{};
    for (int n = 0; n <= MAX; ++n) {
        c[n][0] = 1;
        c[n][n] = 1;
        for (int k = 1; k < n; ++k)
            c[n][k] = c[n - 1][k - 1] + c[n - 1][k];
    }
    return c;
}

constexpr int row_base(int n) {
    const int m = n >> 1;
    return (n & 1) ? (m + 1) * (m + 1) : m * (m + 1);
}

std::uint32_t compact_choose(int n, int k) {
    if (n < 0 || n > MAX || k < 0 || k > n) return 0;
    if (k > n - k) k = n - k;
    return COMPACT[std::size_t(row_base(n) + k)];
}
}

int main() {
    const auto choose = build_choose();
    std::uint64_t cases = 0;
    std::uint32_t max_value = 0;
    int expected_base = 0;
    for (int n = 0; n <= MAX; ++n) {
        const int base = row_base(n);
        if (base != expected_base) {
            std::cerr << "row base mismatch n=" << n << " got=" << base
                      << " expected=" << expected_base << '\n';
            return 2;
        }
        expected_base += n / 2 + 1;
        for (int k = -1; k <= n + 1; ++k) {
            const Rank64 ref = (k < 0 || k > n) ? 0 : choose[n][k];
            const std::uint32_t got = compact_choose(n, k);
            if (ref != got) {
                std::cerr << "choose mismatch n=" << n << " k=" << k
                          << " ref=" << ref << " got=" << got << '\n';
                return 3;
            }
            if (k >= 0 && k <= n) {
                ++cases;
                if (ref > 0xffffffffULL) return 4;
                if (got > max_value) max_value = got;
            }
        }
    }
    if (expected_base != ENTRIES) return 5;
    if (cases != 435ULL) return 6;
    if (max_value != 40116600u) return 7;
    constexpr std::uint64_t old_bytes = 29ULL * 29ULL * 8ULL;
    constexpr std::uint64_t new_bytes = ENTRIES * 4ULL;
    static_assert(old_bytes == 6728);
    static_assert(new_bytes == 900);
    std::cout << "gridfp-choose-sym-u32-table-proof OK"
              << " production_n_max=28"
              << " valid_choose_cases=" << cases
              << " compact_entries=" << ENTRIES
              << " old_bytes=" << old_bytes
              << " compact_bytes=" << new_bytes
              << " saved_bytes=" << (old_bytes - new_bytes)
              << " max_value=" << max_value
              << " symmetry_exact=1 row_base_exact=1 uint32_exact=1 exact=1\n";
    return 0;
}
