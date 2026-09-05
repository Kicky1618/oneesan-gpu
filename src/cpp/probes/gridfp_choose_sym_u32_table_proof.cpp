#include <array>
#include <cstdint>
#include <iostream>

namespace {
using Rank64 = std::uint64_t;
static constexpr int MAX = 28;
static constexpr int SYM_ENTRIES = 225;
static constexpr int TRI_ENTRIES = 435;
static constexpr int FULL_ENTRIES = (MAX + 1) * (MAX + 1);
static constexpr std::array<std::uint32_t, SYM_ENTRIES> SYM = {
#include "../../cuda/gridfp/gridfp_reduced_production_choose_sym_u32_values.inc"
};
static constexpr std::array<std::uint32_t, TRI_ENTRIES> TRI = {
#include "../../cuda/gridfp/gridfp_reduced_production_choose_tri_u32_values.inc"
};
static constexpr std::array<std::uint32_t, FULL_ENTRIES> FULL = {
#include "../../cuda/gridfp/gridfp_reduced_production_choose_full_u32_values.inc"
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

constexpr int sym_row_base(int n) {
    const int m = n >> 1;
    return (n & 1) ? (m + 1) * (m + 1) : m * (m + 1);
}
constexpr int tri_row_base(int n) { return n * (n + 1) / 2; }

std::uint32_t sym_choose(int n, int k) {
    if (n < 0 || n > MAX || k < 0 || k > n) return 0;
    if (k > n - k) k = n - k;
    return SYM[std::size_t(sym_row_base(n) + k)];
}
std::uint32_t tri_choose(int n, int k) {
    if (n < 0 || n > MAX || k < 0 || k > n) return 0;
    return TRI[std::size_t(tri_row_base(n) + k)];
}
std::uint32_t full_choose(int n, int k) {
    if (n < 0 || n > MAX || k < 0 || k > n) return 0;
    return FULL[std::size_t(n * (MAX + 1) + k)];
}
}

int main() {
    const auto choose = build_choose();
    std::uint64_t cases = 0;
    std::uint32_t max_value = 0;
    int expected_sym_base = 0;
    int expected_tri_base = 0;
    for (int n = 0; n <= MAX; ++n) {
        const int sb = sym_row_base(n);
        const int tb = tri_row_base(n);
        if (sb != expected_sym_base || tb != expected_tri_base) {
            std::cerr << "row base mismatch n=" << n
                      << " sym=" << sb << "/" << expected_sym_base
                      << " tri=" << tb << "/" << expected_tri_base << '\n';
            return 2;
        }
        expected_sym_base += n / 2 + 1;
        expected_tri_base += n + 1;
        for (int k = -1; k <= n + 1; ++k) {
            const Rank64 ref = (k < 0 || k > n) ? 0 : choose[n][k];
            const std::uint32_t sym = sym_choose(n, k);
            const std::uint32_t tri = tri_choose(n, k);
            const std::uint32_t full = full_choose(n, k);
            if (ref != sym || ref != tri || ref != full) {
                std::cerr << "choose mismatch n=" << n << " k=" << k
                          << " ref=" << ref << " sym=" << sym
                          << " tri=" << tri << " full=" << full << '\n';
                return 3;
            }
            if (k >= 0 && k <= n) {
                ++cases;
                if (ref > 0xffffffffULL) return 4;
                if (sym > max_value) max_value = sym;
            }
        }
        for (int k = n + 1; k <= MAX; ++k) {
            if (FULL[std::size_t(n * (MAX + 1) + k)] != 0) {
                std::cerr << "full padding nonzero n=" << n << " k=" << k << '\n';
                return 8;
            }
        }
    }
    if (expected_sym_base != SYM_ENTRIES || expected_tri_base != TRI_ENTRIES) return 5;
    if (cases != 435ULL) return 6;
    if (max_value != 40116600u) return 7;
    constexpr std::uint64_t old_bytes = 29ULL * 29ULL * 8ULL;
    constexpr std::uint64_t sym_bytes = SYM_ENTRIES * 4ULL;
    constexpr std::uint64_t tri_bytes = TRI_ENTRIES * 4ULL;
    constexpr std::uint64_t full_bytes = FULL_ENTRIES * 4ULL;
    static_assert(old_bytes == 6728);
    static_assert(sym_bytes == 900);
    static_assert(tri_bytes == 1740);
    static_assert(full_bytes == 3364);
    std::cout << "gridfp-choose-sym-u32-table-proof OK"
              << " production_n_max=28"
              << " valid_choose_cases=" << cases
              << " sym_entries=" << SYM_ENTRIES
              << " tri_entries=" << TRI_ENTRIES
              << " full_entries=" << FULL_ENTRIES
              << " old_bytes=" << old_bytes
              << " sym_bytes=" << sym_bytes
              << " tri_bytes=" << tri_bytes
              << " full_bytes=" << full_bytes
              << " sym_saved_bytes=" << (old_bytes - sym_bytes)
              << " tri_saved_bytes=" << (old_bytes - tri_bytes)
              << " full_saved_bytes=" << (old_bytes - full_bytes)
              << " max_value=" << max_value
              << " symmetry_exact=1 triangle_exact=1 full_shape_exact=1"
              << " row_base_exact=1 uint32_exact=1 exact=1\n";
    return 0;
}
