#include <array>
#include <cstdint>
#include <iostream>
#include <limits>

namespace {

constexpr std::uint32_t P = 4294967291u; // 2^32 - 5
constexpr int MAX_PAIRS = 20;
constexpr int MAX_COEFFICIENT = 2;
constexpr std::uint64_t MAX_MAG =
    std::uint64_t(MAX_PAIRS) * MAX_COEFFICIENT * (std::uint64_t(P) - 1u);
constexpr std::uint64_t MAX_HI = MAX_MAG >> 32;
constexpr std::uint64_t MAX_FOLDED =
    std::uint64_t(std::numeric_limits<std::uint32_t>::max()) + 5ULL * MAX_HI;

static_assert(P == (std::uint64_t(1) << 32) - 5u);
static_assert(MAX_MAG == 171798691600ULL);
static_assert(MAX_HI == 39u);
static_assert(MAX_FOLDED == 4294967490ULL);
static_assert(MAX_FOLDED < 2ULL * std::uint64_t(P));

std::uint32_t reduce_p32m5(long long accum) {
    const bool negative = accum < 0;
    const std::uint64_t magnitude = negative
        ? std::uint64_t(-(accum + 1)) + 1u
        : std::uint64_t(accum);
    std::uint64_t folded =
        std::uint64_t(std::uint32_t(magnitude)) + 5ULL * (magnitude >> 32);
    if (folded >= P) folded -= P;
    const std::uint32_t residue = std::uint32_t(folded);
    return negative && residue ? P - residue : residue;
}

std::uint32_t reference(long long accum) {
    long long residue = accum % static_cast<long long>(P);
    if (residue < 0) residue += P;
    return static_cast<std::uint32_t>(residue);
}

bool check(long long accum, std::uint64_t& cases) {
    const std::uint32_t got = reduce_p32m5(accum);
    const std::uint32_t expected = reference(accum);
    if (got != expected) {
        std::cerr << "p32m5 reduction mismatch accum=" << accum
                  << " got=" << got << " expected=" << expected << '\n';
        return false;
    }
    ++cases;
    return true;
}

} // namespace

int main() {
    std::uint64_t exact_cases = 0;
    std::uint64_t magnitude_cases = 0;

    // For every reachable high 32-bit word, exercise both ends of the low word
    // and the exact boundary where lo+5*hi crosses p. The static assertions
    // above prove that one subtraction is sufficient over the whole range.
    for (std::uint64_t hi = 0; hi <= MAX_HI; ++hi) {
        std::array<std::uint64_t, 5> low{};
        std::size_t n = 0;
        auto add = [&](std::uint64_t x) {
            if (x > std::numeric_limits<std::uint32_t>::max()) return;
            for (std::size_t i = 0; i < n; ++i) if (low[i] == x) return;
            low[n++] = x;
        };

        add(0);
        add(std::numeric_limits<std::uint32_t>::max());
        const std::uint64_t threshold = std::uint64_t(P) - 5ULL * hi;
        if (threshold) add(threshold - 1u);
        add(threshold);
        add(threshold + 1u);

        for (std::size_t i = 0; i < n; ++i) {
            const std::uint64_t magnitude = (hi << 32) | low[i];
            if (magnitude > MAX_MAG) continue;
            ++magnitude_cases;
            if (!check(static_cast<long long>(magnitude), exact_cases)) return 2;
            if (magnitude &&
                !check(-static_cast<long long>(magnitude), exact_cases)) return 3;
        }
    }

    if (magnitude_cases != 196 || exact_cases != 391) {
        std::cerr << "unexpected p32m5 proof coverage magnitudes=" << magnitude_cases
                  << " exact=" << exact_cases << '\n';
        return 4;
    }

    // Extremal production accumulators are included explicitly for readability.
    if (reduce_p32m5(static_cast<long long>(MAX_MAG)) != reference(static_cast<long long>(MAX_MAG)) ||
        reduce_p32m5(-static_cast<long long>(MAX_MAG)) != reference(-static_cast<long long>(MAX_MAG)))
        return 5;

    std::cout << "gridfp-runtime-p32m5-mod-proof OK"
              << " modulus=" << P
              << " max_acc_mag=" << MAX_MAG
              << " max_hi=" << MAX_HI
              << " max_folded=" << MAX_FOLDED
              << " magnitude_cases=" << magnitude_cases
              << " exact_cases=" << exact_cases
              << " one_subtraction_bound=1"
              << " signed_exact=1 division_free_fast_path=1\n";
    return 0;
}
