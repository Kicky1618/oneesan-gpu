#include <array>
#include <cstdint>
#include <iostream>

namespace {
using U64 = std::uint64_t;
constexpr int MAX = 28;
constexpr int COLS = 30;
struct Stats { std::uint64_t nonzero = 0; U64 max_value = 0; };
Stats primitive_stats() {
    std::array<std::array<U64, COLS>, MAX + 1> p{}; p[0][0] = 1;
    for (int rem = 1; rem <= MAX; ++rem) for (int h = 0; h <= MAX; ++h) {
        U64 z = p[rem - 1][h + 1]; if (h > 0) z += p[rem - 1][h - 1]; p[rem][h] = z;
    }
    Stats s{}; for (const auto& row : p) for (U64 v : row) if (v) { ++s.nonzero; if (v > s.max_value) s.max_value = v; } return s;
}
Stats motzkin_stats() {
    std::array<std::array<U64, COLS>, MAX + 1> m{}; m[0][0] = 1;
    for (int rem = 1; rem <= MAX; ++rem) for (int h = 0; h <= MAX; ++h) {
        U64 z = m[rem - 1][h]; if (h > 0) z += m[rem - 1][h - 1]; z += m[rem - 1][h + 1]; m[rem][h] = z;
    }
    Stats s{}; for (const auto& row : m) for (U64 v : row) if (v) { ++s.nonzero; if (v > s.max_value) s.max_value = v; } return s;
}
U64 choose_max() {
    std::array<std::array<U64, MAX + 1>, MAX + 1> c{}; U64 mx = 0;
    for (int n = 0; n <= MAX; ++n) { c[n][0] = c[n][n] = 1; for (int k = 1; k < n; ++k) c[n][k] = c[n - 1][k - 1] + c[n - 1][k]; for (int k = 0; k <= n; ++k) if (c[n][k] > mx) mx = c[n][k]; }
    return mx;
}
}

int main() {
    const Stats primitive = primitive_stats(); const Stats motzkin = motzkin_stats(); const U64 cmax = choose_max();
    if (primitive.nonzero != 225 || primitive.max_value != 8947575ULL) return 2;
    if (motzkin.nonzero != 435 || motzkin.max_value != 569371325796ULL) return 3;
    if (cmax != 40116600ULL) return 4;
    if (cmax > 0xffffffffULL || primitive.max_value > 0xffffffffULL) return 5;
    if (motzkin.max_value <= 0xffffffffULL) return 6;

    constexpr U64 choose_old = 29ULL * 29ULL * 8ULL;
    constexpr U64 primitive_old = 29ULL * 30ULL * 8ULL;
    constexpr U64 motzkin_old = 29ULL * 30ULL * 8ULL;
    constexpr U64 old_total = choose_old + primitive_old + motzkin_old;

    constexpr U64 choose_sym = 225ULL * 4ULL;
    constexpr U64 choose_tri = 435ULL * 4ULL;
    constexpr U64 choose_full_u32 = 29ULL * 29ULL * 4ULL;
    constexpr U64 primitive_compact = 225ULL * 4ULL;
    constexpr U64 primitive_full_u32 = 29ULL * 30ULL * 4ULL;
    constexpr U64 motzkin_tri = 435ULL * 8ULL;

    constexpr U64 max_compact_total = choose_sym + primitive_compact + motzkin_tri;
    constexpr U64 tri_compact_total = choose_tri + primitive_compact + motzkin_tri;
    constexpr U64 low_arith_total = choose_full_u32 + primitive_full_u32 + motzkin_tri;

    static_assert(old_total == 20648);
    static_assert(max_compact_total == 5280);
    static_assert(tri_compact_total == 6120);
    static_assert(low_arith_total == 10324);
    static_assert(old_total - max_compact_total == 15368);
    static_assert(old_total - tri_compact_total == 14528);
    static_assert(old_total - low_arith_total == 10324);

    std::cout << "gridfp-codec-table-budget-proof OK"
              << " choose_max=" << cmax
              << " primitive_nonzero=" << primitive.nonzero
              << " primitive_max=" << primitive.max_value
              << " motzkin_nonzero=" << motzkin.nonzero
              << " motzkin_max=" << motzkin.max_value
              << " old_total_bytes=" << old_total
              << " max_compact_bytes=" << max_compact_total
              << " max_compact_saved_bytes=" << (old_total - max_compact_total)
              << " tri_compact_bytes=" << tri_compact_total
              << " tri_compact_saved_bytes=" << (old_total - tri_compact_total)
              << " low_arith_bytes=" << low_arith_total
              << " low_arith_saved_bytes=" << (old_total - low_arith_total)
              << " choose_full_u32_bytes=" << choose_full_u32
              << " primitive_full_u32_bytes=" << primitive_full_u32
              << " motzkin_tri_bytes=" << motzkin_tri
              << " choose_u32=1 primitive_u32=1 motzkin_u32=0 exact=1\n";
    return 0;
}
