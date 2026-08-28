#include <cstdint>
#include <iostream>

namespace {

std::uint64_t binom(int n, int k) {
    if (n < 0 || k < 0 || k > n) return 0;
    std::uint64_t x = 1;
    for (int i = 1; i <= k; ++i)
        x = x * std::uint64_t(n - k + i) / std::uint64_t(i);
    return x;
}

std::uint64_t old_rank(std::uint32_t mask, int len, int ones) {
    std::uint64_t rank = 0;
    int left = ones;
    for (int pos = 0; pos < len; ++pos) {
        if (((mask >> pos) & 1u) == 0) continue;
        rank += binom(len - pos - 1, left);
        --left;
    }
    return rank;
}

std::uint64_t setbit_rank(std::uint32_t mask, int len, int ones) {
    if (len < 32) mask &= (std::uint32_t(1) << len) - 1u;
    std::uint64_t rank = 0;
    int left = ones;
    while (mask) {
        const int pos = __builtin_ctz(mask);
        rank += binom(len - pos - 1, left);
        --left;
        mask &= mask - 1;
    }
    return rank;
}

bool check(std::uint32_t mask, int len, std::uint64_t& cases) {
    const std::uint32_t clipped = len < 32 ? mask & ((std::uint32_t(1) << len) - 1u) : mask;
    const int ones = __builtin_popcount(clipped);
    const auto a = old_rank(clipped, len, ones);
    const auto b = setbit_rank(clipped, len, ones);
    if (a != b) {
        std::cerr << "support setbit mismatch mask=" << mask << " len=" << len
                  << " ones=" << ones << " old=" << a << " setbit=" << b << '\n';
        return false;
    }
    ++cases;
    return true;
}

} // namespace

int main() {
    std::uint64_t exhaustive_cases = 0;
    for (int len = 0; len <= 16; ++len) {
        const std::uint32_t end = std::uint32_t(1) << len;
        for (std::uint32_t mask = 0; mask < end; ++mask)
            if (!check(mask, len, exhaustive_cases)) return 2;
    }

    std::uint64_t random_cases = 0;
    std::uint64_t s = 0x13198a2e03707344ULL;
    for (std::uint64_t tc = 0; tc < 1000000ULL; ++tc) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        const int len = int((s >> 59) % 29);
        if (!check(static_cast<std::uint32_t>(s), len, random_cases)) return 3;
    }

    std::cout << "gridfp-runtime-support-rank-setbits-proof OK"
              << " exhaustive_len_max=16"
              << " exhaustive_cases=" << exhaustive_cases
              << " random_cases=" << random_cases
              << " runtime_len_max=28"
              << " old_scan=len new_scan=ones exact=1\n";
    return 0;
}
