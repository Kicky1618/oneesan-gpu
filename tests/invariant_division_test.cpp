#include "../src/common/invariant_division.hpp"
#include <cstdlib>
#include <iostream>
#include <random>
#include <set>

static uint64_t checked = 0;
static void check(uint64_t n, uint32_t d) {
    const auto got = oneesan::invariant_divmod(n, d, oneesan::division_reciprocal(d));
    if (got.quotient != n / d || got.remainder != n % d) {
        std::cerr << "divmod mismatch n=" << n << " d=" << d << '\n';
        std::exit(1);
    }
    ++checked;
}

static void boundaries(uint32_t d) {
    const uint64_t max = UINT64_MAX;
    for (uint64_t q : {uint64_t(0), uint64_t(1), uint64_t(UINT32_MAX), max / d}) {
        const uint64_t n = q * d;
        check(n, d);
        if (n) check(n - 1, d);
        if (n < max) check(n + 1, d);
    }
    check(max, d);
    check(max - 1, d);
}

int main() {
    for (uint32_t d = 1; d <= 256; ++d)
        for (uint64_t n = 0; n < 4096; ++n) check(n, d);
    for (uint32_t d = 1; d <= 65536; ++d) boundaries(d);
    for (unsigned bit = 0; bit < 32; ++bit) {
        const uint32_t d = uint32_t(1) << bit;
        boundaries(d);
        if (d > 1) boundaries(d - 1);
        boundaries(d + 1);
    }
    boundaries(UINT32_MAX);

    // All possible LOW14 occupancy masks and start heights. These are the
    // production factor strides; generate them independently with walk DP.
    std::set<uint32_t> strides;
    for (uint32_t mask = 0; mask < (1u << 14); ++mask) {
        uint32_t dp[15][16]{};
        dp[0][0] = 1;
        for (int w = 1; w <= 14; ++w)
            for (int h = 0; h <= 14; ++h)
                dp[w][h] = (mask & (1u << (w - 1)))
                    ? (h ? dp[w-1][h-1] : 0) + dp[w-1][h+1]
                    : dp[w-1][h];
        for (int h = 0; h <= 14; ++h)
            if (dp[14][h]) strides.insert(dp[14][h]);
    }
    uint32_t free_dp[15][16]{};
    free_dp[0][0] = 1;
    for (int w = 1; w <= 14; ++w)
        for (int h = 0; h <= 14; ++h)
            free_dp[w][h] = free_dp[w-1][h]
                + (h ? free_dp[w-1][h-1] : 0) + free_dp[w-1][h+1];
    for (int h = 0; h <= 14; ++h)
        if (free_dp[14][h]) strides.insert(free_dp[14][h]);
    std::mt19937_64 rng(0x6f6e656573616eULL);
    for (uint32_t d : strides) {
        boundaries(d);
        for (int j = 0; j < 10000; ++j) check(rng(), d);
    }
    for (int i = 0; i < 1000000; ++i) {
        uint32_t d = uint32_t(rng());
        if (d) check(rng(), d);
    }
    std::cout << "PASS " << checked << " divisions; " << strides.size()
              << " production strides\n";
}
