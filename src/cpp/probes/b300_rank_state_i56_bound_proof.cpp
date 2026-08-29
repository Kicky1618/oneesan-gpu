#include "../../common/gridfp_transition.hpp"

#include <cstdint>
#include <iostream>

using Code = std::uint64_t;
using Delta = std::int64_t;

namespace {
constexpr int MAXW = 28;
constexpr std::uint64_t DELTA_MASK = (std::uint64_t(1) << 56) - 1;
constexpr std::uint64_t DELTA_SIGN = std::uint64_t(1) << 55;
constexpr std::uint64_t DELTA_MOD = std::uint64_t(1) << 56;
constexpr Delta DELTA_MIN = -(Delta(1) << 55);
constexpr Delta DELTA_MAX = (Delta(1) << 55) - 1;

std::uint64_t full_count(int W) {
    std::uint64_t dp[MAXW + 1][MAXW + 3]{};
    for (int h = 0; h <= MAXW + 2; ++h) dp[0][h] = (h == 0);
    for (int w = 1; w <= W; ++w) {
        for (int h = 0; h <= MAXW + 1; ++h) {
            std::uint64_t x = dp[w - 1][h];
            if (h > 0) x += dp[w - 1][h - 1];
            if (h < MAXW + 1) x += dp[w - 1][h + 1];
            dp[w][h] = x;
        }
    }
    return dp[W][1];
}

std::uint64_t pack(Delta d, std::uint8_t h) {
    return (std::uint64_t(d) & DELTA_MASK) | (std::uint64_t(h) << 56);
}
Delta unpack_delta(std::uint64_t s) {
    const std::uint64_t raw = s & DELTA_MASK;
    return (raw & DELTA_SIGN) ? Delta(raw - DELTA_MOD) : Delta(raw);
}
std::uint8_t unpack_height(std::uint64_t s) { return std::uint8_t(s >> 56); }
}

int main() {
    const Code max_full = full_count(MAXW);
    if (max_full > Code(DELTA_MAX)) return 2;
    if (max_full != 385719506620ULL) return 3;

    const Delta probes[] = {
        0, 1, -1, Delta(max_full), -Delta(max_full),
        DELTA_MAX, DELTA_MIN
    };
    std::uint64_t roundtrips = 0;
    for (Delta d : probes) {
        for (int h = 0; h <= 28; ++h) {
            const auto s = pack(d, std::uint8_t(h));
            if (unpack_delta(s) != d || unpack_height(s) != h) return 4;
            ++roundtrips;
        }
    }
    std::cout << "b300-rank-state-i56-bound-proof OK width_max=28"
              << " full_state_bound=" << max_full
              << " signed_delta_bits=56 height_bits=8 storage_bytes=8"
              << " signed_decode=defined_subtract"
              << " fallback_required=0 roundtrips=" << roundtrips
              << " exact=1\n";
    return 0;
}
