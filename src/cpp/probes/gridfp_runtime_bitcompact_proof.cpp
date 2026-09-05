#include <cstdint>
#include <iostream>
#include <random>

namespace {

std::uint32_t compact_outside_ref(
    std::uint32_t full, int W, int lo, int hi
) {
    std::uint32_t compact = 0;
    int q = 0;
    for (int bit = 0; bit < W; ++bit) {
        if (bit >= lo && bit <= hi) continue;
        if ((full >> bit) & 1u) compact |= std::uint32_t(1) << q;
        ++q;
    }
    return compact;
}

std::uint32_t compact_outside_fast(
    std::uint32_t full, int W, int lo, int hi
) {
    const std::uint32_t width_mask = W == 32 ? ~0u :
        ((std::uint32_t(1) << W) - 1u);
    full &= width_mask;
    const std::uint32_t low_mask = lo ?
        ((std::uint32_t(1) << lo) - 1u) : 0u;
    const std::uint32_t low = full & low_mask;
    const std::uint32_t high = full >> (hi + 1);
    return low | (high << lo);
}

std::uint32_t erase_two_ref(
    std::uint32_t local, int L, int a, int b
) {
    std::uint32_t compact = 0;
    int q = 0;
    for (int pos = 0; pos < L; ++pos) {
        if (pos == a || pos == b) continue;
        if ((local >> pos) & 1u) compact |= std::uint32_t(1) << q;
        ++q;
    }
    return compact;
}

std::uint32_t erase_two_fast(
    std::uint32_t local, int L, int a, int b
) {
    const int lo = a < b ? a : b;
    const int hi = a < b ? b : a;
    const std::uint32_t width_mask = L == 32 ? ~0u :
        ((std::uint32_t(1) << L) - 1u);
    local &= width_mask;
    const std::uint32_t low_mask = lo ?
        ((std::uint32_t(1) << lo) - 1u) : 0u;
    const int middle_width = hi - lo - 1;
    const std::uint32_t middle_mask = middle_width ?
        ((std::uint32_t(1) << middle_width) - 1u) : 0u;
    const std::uint32_t low = local & low_mask;
    const std::uint32_t middle = (local >> (lo + 1)) & middle_mask;
    const std::uint32_t high = local >> (hi + 1);
    return low | (middle << lo) | (high << (hi - 1));
}

} // namespace

int main() {
    std::uint64_t exhaustive_outside = 0;
    std::uint64_t exhaustive_erase = 0;
    for (int W = 1; W <= 12; ++W) {
        for (int lo = 0; lo < W; ++lo) {
            for (int hi = lo; hi < W; ++hi) {
                const std::uint32_t limit = std::uint32_t(1) << W;
                for (std::uint32_t full = 0; full < limit; ++full) {
                    if (compact_outside_ref(full, W, lo, hi) !=
                        compact_outside_fast(full, W, lo, hi)) {
                        std::cerr << "outside mismatch W=" << W
                                  << " lo=" << lo << " hi=" << hi
                                  << " full=" << full << '\n';
                        return 2;
                    }
                    ++exhaustive_outside;
                }
            }
        }
    }
    for (int L = 2; L <= 12; ++L) {
        for (int a = 0; a < L; ++a) {
            for (int b = 0; b < L; ++b) {
                if (a == b) continue;
                const std::uint32_t limit = std::uint32_t(1) << L;
                for (std::uint32_t local = 0; local < limit; ++local) {
                    if (erase_two_ref(local, L, a, b) !=
                        erase_two_fast(local, L, a, b)) {
                        std::cerr << "erase mismatch L=" << L
                                  << " a=" << a << " b=" << b
                                  << " local=" << local << '\n';
                        return 3;
                    }
                    ++exhaustive_erase;
                }
            }
        }
    }

    std::mt19937_64 rng(0x5eedf00dULL);
    constexpr std::uint64_t RANDOM_CASES = 2000000;
    for (std::uint64_t i = 0; i < RANDOM_CASES; ++i) {
        const int W = 1 + int(rng() % 28);
        const int lo = int(rng() % W);
        const int hi = lo + int(rng() % (W - lo));
        const std::uint32_t full = std::uint32_t(rng()) &
            ((std::uint32_t(1) << W) - 1u);
        if (compact_outside_ref(full, W, lo, hi) !=
            compact_outside_fast(full, W, lo, hi)) return 4;

        const int L = 2 + int(rng() % 14); // production local width <=15
        const int a = int(rng() % L);
        int b = int(rng() % (L - 1));
        if (b >= a) ++b;
        const std::uint32_t local = std::uint32_t(rng()) &
            ((std::uint32_t(1) << L) - 1u);
        if (erase_two_ref(local, L, a, b) !=
            erase_two_fast(local, L, a, b)) return 5;
    }

    std::cout << "gridfp-runtime-bitcompact-proof OK"
              << " exhaustive_max_bits=12"
              << " exhaustive_outside=" << exhaustive_outside
              << " exhaustive_erase=" << exhaustive_erase
              << " random_cases=" << RANDOM_CASES
              << " production_W_max=28 production_L_max=15"
              << " outside_loop_iterations_removed=W"
              << " erase_loop_iterations_removed=L"
              << " exact=1\n";
    return 0;
}
