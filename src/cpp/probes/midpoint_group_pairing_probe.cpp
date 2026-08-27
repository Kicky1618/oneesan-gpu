#include <array>
#include <cstdint>
#include <iostream>

static constexpr int W = 28;
static constexpr int L = 14;
static constexpr int H = 13;

static std::uint32_t revn(std::uint32_t x, int n) {
    std::uint32_t z = 0;
    for (int i = 0; i < n; ++i) if ((x >> i) & 1u) z |= 1u << (n - 1 - i);
    return z;
}

static std::uint32_t rev28(std::uint32_t x) { return revn(x, 28); }

int main() {
    std::uint64_t odd_masks = 0;
    std::uint64_t representatives = 0;
    std::array<std::uint64_t, 29> by_m{};
    std::array<std::uint64_t, 29> rep_by_m{};

    for (std::uint32_t right_high = 0; right_high < (1u << H); ++right_high) {
        std::uint32_t left_low13 = revn(right_high, H);
        for (int left_bit13 = 0; left_bit13 <= 1; ++left_bit13) {
            std::uint32_t left_low = left_low13 | (std::uint32_t(left_bit13) << 13);
            int kl = __builtin_popcount(left_low);

            for (std::uint32_t left_high = 0; left_high < (1u << H); ++left_high) {
                int kh = __builtin_popcount(left_high);
                // Exactly one center occupancy makes the total occupancy odd.
                int left_center = ((kl + kh) & 1) ? 0 : 1;
                int m = kl + kh + left_center;

                std::uint32_t s = left_low
                    | (std::uint32_t(left_center) << 14)
                    | (left_high << 15);
                std::uint32_t t = rev28(s);
                if ((__builtin_popcount(s) & 1) == 0) {
                    std::cerr << "constructed even occupancy\n";
                    return 1;
                }
                ++odd_masks;
                ++by_m[m];

                std::uint32_t got_right_high = (t >> 15) & ((1u << 13) - 1u);
                int got_right_center = (t >> 14) & 1u;
                std::uint32_t got_right_low = t & ((1u << 14) - 1u);
                std::uint32_t want_right_low = revn(left_high, H)
                    | (std::uint32_t(left_center) << 13);

                if (got_right_high != right_high
                    || got_right_center != left_bit13
                    || got_right_low != want_right_low) {
                    std::cerr << "reflection decomposition mismatch"
                              << " right_high=" << right_high
                              << " left_bit13=" << left_bit13
                              << " left_high=" << left_high << "\n";
                    return 2;
                }

                // Odd occupancy cannot be reflection-fixed at even width 28.
                if (s == t) {
                    std::cerr << "unexpected reflection-fixed odd mask s=" << s << "\n";
                    return 3;
                }
                if (s < t) {
                    ++representatives;
                    ++rep_by_m[m];
                }
            }
        }
    }

    std::cout << "midpoint_group_pairing"
              << " odd_masks=" << odd_masks
              << " representatives=" << representatives
              << " high_groups=" << (1u << H)
              << " low_groups=" << (1u << L) << "\n";

    if (odd_masks != (1ull << 27) || representatives != (1ull << 26)) {
        std::cerr << "unexpected mask cardinality\n";
        return 4;
    }

    for (int m = 1; m <= 27; m += 2) {
        if (rep_by_m[m] * 2 != by_m[m]) {
            std::cerr << "m distribution is not paired m=" << m
                      << " all=" << by_m[m] << " rep=" << rep_by_m[m] << "\n";
            return 5;
        }
        std::cout << "m=" << m
                  << " masks=" << by_m[m]
                  << " representative_pairs=" << rep_by_m[m] << "\n";
    }

    std::cout << "PASS LOW14/HIGH13 reflection pairing\n";
    return 0;
}
