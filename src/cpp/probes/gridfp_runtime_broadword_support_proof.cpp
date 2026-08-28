#include <cstdint>
#include <iostream>

namespace {

std::uint32_t reference_support(std::uint64_t mate, int len) {
    std::uint32_t mask = 0;
    for (int bit = 0; bit < len; ++bit)
        if (((mate >> (2 * bit)) & 3ULL) != 0)
            mask |= std::uint32_t(1) << bit;
    return mask;
}

std::uint32_t broadword_support(std::uint64_t mate, int len) {
    std::uint64_t x = (mate | (mate >> 1)) & 0x5555555555555555ULL;
    x = (x | (x >> 1)) & 0x3333333333333333ULL;
    x = (x | (x >> 2)) & 0x0f0f0f0f0f0f0f0fULL;
    x = (x | (x >> 4)) & 0x00ff00ff00ff00ffULL;
    x = (x | (x >> 8)) & 0x0000ffff0000ffffULL;
    x = (x | (x >> 16)) & 0x00000000ffffffffULL;
    std::uint32_t out = static_cast<std::uint32_t>(x);
    if (len < 32) out &= (std::uint32_t(1) << len) - 1u;
    return out;
}

bool check(std::uint64_t mate, int len, std::uint64_t& cases) {
    const auto a = reference_support(mate, len);
    const auto b = broadword_support(mate, len);
    if (a != b) {
        std::cerr << "broadword support mismatch mate=" << mate
                  << " len=" << len << " reference=" << a
                  << " broadword=" << b << '\n';
        return false;
    }
    ++cases;
    return true;
}

} // namespace

int main() {
    std::uint64_t exhaustive_cases = 0;
    for (std::uint64_t mate = 0; mate < (1ULL << 16); ++mate)
        for (int len = 0; len <= 8; ++len)
            if (!check(mate, len, exhaustive_cases)) return 2;

    std::uint64_t random_cases = 0;
    std::uint64_t s = 0x243f6a8885a308d3ULL;
    for (std::uint64_t tc = 0; tc < 1000000ULL; ++tc) {
        s ^= s << 7;
        s ^= s >> 9;
        s ^= s << 8;
        const std::uint64_t mate = s;
        const int len = int((s >> 58) % 29);
        if (!check(mate, len, random_cases)) return 3;
    }

    std::cout << "gridfp-runtime-broadword-support-proof OK"
              << " exhaustive_pair_bits=8"
              << " exhaustive_cases=" << exhaustive_cases
              << " random_cases=" << random_cases
              << " max_runtime_len=28"
              << " broadword_stages=6"
              << " exact=1\n";
    return 0;
}
