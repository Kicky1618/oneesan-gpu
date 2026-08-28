#include <array>
#include <cstdint>
#include <iostream>
#include <set>

namespace {

constexpr uint32_t kChunk = 5;
constexpr uint32_t kKeys = 243;
constexpr uint32_t kStates = 26;

constexpr uint32_t pow3(uint32_t n) {
    return n == 0 ? 1u : 3u * pow3(n - 1);
}

constexpr uint8_t lmask_host(uint32_t key) {
    uint8_t mask = 0;
    for (uint32_t pos = 0; pos < kChunk; ++pos) {
        const uint32_t v = (key / pow3(pos)) % 3u;
        if (v == 1u) mask = uint8_t(mask | uint8_t(1u << pos));
    }
    return mask;
}

constexpr uint8_t popcount5(uint8_t x) {
    uint8_t n = 0;
    for (uint32_t i = 0; i < kChunk; ++i) n = uint8_t(n + ((x >> i) & 1u));
    return n;
}

constexpr uint8_t ordinary_mask_host(uint32_t key, uint32_t input_state) {
    uint32_t s = input_state;
    uint8_t mask = 0;
    for (int pos = int(kChunk) - 1; pos >= 0; --pos) {
        const uint32_t v = (key / pow3(uint32_t(pos))) % 3u;
        if (v == 2u) {
            if (s == 1u) break;
            --s;
        } else if (v == 1u) {
            if (s == 1u) mask = uint8_t(mask | uint8_t(1u << pos));
            ++s;
        }
    }
    return mask;
}

constexpr uint8_t rankmask_host(uint32_t key, uint32_t input_state) {
    const uint8_t mask = ordinary_mask_host(key, input_state);
    const uint8_t lm = lmask_host(key);
    uint8_t rankmask = 0;
    for (uint32_t pos = 0; pos < kChunk; ++pos) {
        if (((mask >> pos) & 1u) == 0u) continue;
        const uint8_t lower_or_equal = uint8_t((uint8_t(1u << (pos + 1u))) - 1u);
        const uint8_t higher = uint8_t(lm & uint8_t(~lower_or_equal));
        const uint8_t ordinal = popcount5(higher);
        rankmask = uint8_t(rankmask | uint8_t(1u << ordinal));
    }
    return rankmask;
}

uint32_t loop_projection(uint8_t mask) {
    uint32_t packed = 0;
    uint32_t out = 0;
    while (mask) {
        uint32_t ordinal = 0;
        while (((mask >> ordinal) & 1u) == 0u) ++ordinal;
        mask = uint8_t(mask & uint8_t(mask - 1));
        packed |= (ordinal + 1u) << (4u * out++);
    }
    return packed;
}

uint32_t direct3_projection(uint8_t mask) {
    uint32_t packed = 0;
    uint32_t out = 0;
    for (uint32_t ordinal = 0; ordinal < 3; ++ordinal)
        if (mask & uint8_t(1u << ordinal))
            packed |= (ordinal + 1u) << (4u * out++);
    return packed;
}

}  // namespace

int main() {
    std::array<uint64_t, 6> hist{};
    std::set<uint32_t> masks;
    uint64_t cases = 0;
    bool upper_bits_zero = true;
    bool direct3_exact = true;
    uint32_t max_popcount = 0;

    for (uint32_t state = 1; state < kStates; ++state) {
        for (uint32_t key = 0; key < kKeys; ++key) {
            const uint8_t mask = rankmask_host(key, state);
            const uint32_t pc = popcount5(mask);
            ++hist[pc];
            masks.insert(mask);
            ++cases;
            if (mask & 0x18u) upper_bits_zero = false;
            if (loop_projection(mask) != direct3_projection(mask)) direct3_exact = false;
            if (pc > max_popcount) max_popcount = pc;
        }
    }

    const std::set<uint32_t> expected{0u, 1u, 2u, 3u, 5u, 7u};
    const bool exact_mask_set = masks == expected;
    const bool exact_hist =
        hist[0] == 5855u && hist[1] == 187u && hist[2] == 32u &&
        hist[3] == 1u && hist[4] == 0u && hist[5] == 0u;

    std::cout << "cross5-rankmask-shape cases=" << cases
              << " mask_set_exact=" << exact_mask_set
              << " upper_bits_zero=" << upper_bits_zero
              << " direct3_exact=" << direct3_exact
              << " max_popcount=" << max_popcount << '\n';
    std::cout << "popcount_hist="
              << hist[0] << ',' << hist[1] << ',' << hist[2] << ','
              << hist[3] << ',' << hist[4] << ',' << hist[5] << '\n';
    std::cout << "allowed_masks=0,1,2,3,5,7\n";

    if (cases != 6075u || !exact_mask_set || !upper_bits_zero ||
        !direct3_exact || max_popcount != 3u || !exact_hist)
        return 1;
    return 0;
}
