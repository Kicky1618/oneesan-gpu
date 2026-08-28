#include <array>
#include <cstdint>
#include <iostream>
#include <limits>

namespace {

struct FastDiv64 {
    std::uint64_t divisor = 1;
    std::uint64_t magic = 0;
};

FastDiv64 make_fastdiv64(std::uint64_t d) {
    if (!d) return FastDiv64{0, 0};
    if (d == 1) return FastDiv64{1, 0};
    return FastDiv64{d, std::numeric_limits<std::uint64_t>::max() / d + 1};
}

std::uint64_t mulhi64(std::uint64_t a, std::uint64_t b) {
    return static_cast<std::uint64_t>((static_cast<__uint128_t>(a) * b) >> 64);
}

void fastdivmod64(
    std::uint64_t x,
    FastDiv64 fd,
    std::uint64_t& q,
    std::uint64_t& r
) {
    if (fd.divisor == 1) {
        q = x;
        r = 0;
        return;
    }
    std::uint64_t q0 = mulhi64(x, fd.magic);
    const __uint128_t p0 = static_cast<__uint128_t>(q0) * fd.divisor;
    const std::uint64_t p0_lo = static_cast<std::uint64_t>(p0);
    const std::uint64_t p0_hi = static_cast<std::uint64_t>(p0 >> 64);
    if (p0_hi || p0_lo > x) --q0;
    q = q0;
    r = x - q * fd.divisor;
}

template<int Bits>
bool exhaustive_small(std::uint64_t& cases, std::uint64_t& corrections) {
    static_assert(Bits > 1 && Bits <= 16);
    const std::uint32_t B = std::uint32_t(1) << Bits;
    const std::uint32_t mask = B - 1;
    for (std::uint32_t d = 1; d < B; ++d) {
        const std::uint32_t magic = d == 1 ? 0u : (mask / d + 1u);
        for (std::uint32_t x = 0; x < B; ++x) {
            std::uint32_t q = 0, r = 0;
            if (d == 1) {
                q = x;
            } else {
                std::uint32_t q0 = (x * magic) >> Bits;
                const std::uint32_t p0 = q0 * d;
                if (p0 >= B || p0 > x) {
                    --q0;
                    ++corrections;
                }
                q = q0;
                r = x - q * d;
            }
            if (q != x / d || r != x % d) {
                std::cerr << "small fastdiv mismatch bits=" << Bits
                          << " x=" << x << " d=" << d
                          << " q=" << q << " refq=" << x / d
                          << " r=" << r << " refr=" << x % d << '\n';
                return false;
            }
            ++cases;
        }
    }
    return true;
}

bool check64(
    std::uint64_t x,
    std::uint64_t d,
    std::uint64_t& cases,
    std::uint64_t& corrections
) {
    if (!d) return false;
    const FastDiv64 fd = make_fastdiv64(d);
    const std::uint64_t q0 = d == 1 ? x : mulhi64(x, fd.magic);
    if (d != 1) {
        const __uint128_t p0 = static_cast<__uint128_t>(q0) * d;
        if ((p0 >> 64) || static_cast<std::uint64_t>(p0) > x) ++corrections;
    }
    std::uint64_t q = 0, r = 0;
    fastdivmod64(x, fd, q, r);
    if (q != x / d || r != x % d || r >= d ||
        static_cast<__uint128_t>(q) * d + r != x) {
        std::cerr << "fastdiv64 mismatch x=" << x << " d=" << d
                  << " q=" << q << " refq=" << x / d
                  << " r=" << r << " refr=" << x % d << '\n';
        return false;
    }
    ++cases;
    return true;
}

} // namespace

int main() {
    std::uint64_t small_cases = 0, small_corrections = 0;
    if (!exhaustive_small<12>(small_cases, small_corrections)) return 2;

    std::uint64_t cases64 = 0, corrections64 = 0;
    constexpr std::uint64_t U = std::numeric_limits<std::uint64_t>::max();
    const std::array<std::uint64_t, 18> divisors = {
        1ULL, 2ULL, 3ULL, 5ULL, 7ULL, 11ULL, 31ULL, 63ULL,
        127ULL, 255ULL, 256ULL, 257ULL, 65535ULL, 65537ULL,
        4294967291ULL, (1ULL << 32), (1ULL << 63) - 1ULL, U};
    const std::array<std::uint64_t, 18> numerators = {
        0ULL, 1ULL, 2ULL, 3ULL, 7ULL, 15ULL, 31ULL, 63ULL,
        255ULL, 256ULL, 65535ULL, 65536ULL, 4294967295ULL,
        473397057701ULL, (1ULL << 48) - 1ULL,
        (1ULL << 63) - 1ULL, (1ULL << 63), U};
    for (const auto d : divisors)
        for (const auto x : numerators)
            if (!check64(x, d, cases64, corrections64)) return 3;

    std::uint64_t s0 = 0x243f6a8885a308d3ULL;
    std::uint64_t s1 = 0x13198a2e03707344ULL;
    for (std::uint64_t i = 0; i < 2000000ULL; ++i) {
        s0 ^= s0 << 7; s0 ^= s0 >> 9; s0 ^= s0 << 8;
        s1 ^= s1 << 13; s1 ^= s1 >> 7; s1 ^= s1 << 17;
        const std::uint64_t x = s0;
        const std::uint64_t d = s1 ? s1 : 1ULL;
        if (!check64(x, d, cases64, corrections64)) return 4;
    }

    // Production-scale numerators are below the W=28 state count. Exercise
    // divisors densely around small values and pseudo-randomly up to that bound.
    constexpr std::uint64_t PROD_MAX = 473397057701ULL;
    for (std::uint64_t d = 1; d <= 65536; ++d) {
        const std::array<std::uint64_t, 6> xs = {
            0ULL, d - 1ULL, d, d < PROD_MAX ? d + 1ULL : PROD_MAX,
            PROD_MAX / 2ULL, PROD_MAX};
        for (auto x : xs) {
            if (x > PROD_MAX) x = PROD_MAX;
            if (!check64(x, d, cases64, corrections64)) return 5;
        }
    }

    std::cout << "gridfp-runtime-fastdiv64-proof OK"
              << " small_bits=12"
              << " small_cases=" << small_cases
              << " small_corrections=" << small_corrections
              << " cases64=" << cases64
              << " corrections64=" << corrections64
              << " random64=2000000"
              << " production_max_numerator=" << PROD_MAX
              << " quotient_error_bound=1"
              << " product_overflow_checked=1"
              << " exact=1\n";
    return 0;
}
