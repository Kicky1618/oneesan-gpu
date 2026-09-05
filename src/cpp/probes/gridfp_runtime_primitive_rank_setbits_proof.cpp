#include <array>
#include <cstdint>
#include <iostream>

namespace {

constexpr int MAX_W = 28;
using Rank64 = std::uint64_t;
using MateID = std::uint64_t;

enum MateValue : std::uint8_t { N = 0, R = 1, L = 2 };

std::array<std::array<Rank64, MAX_W + 2>, MAX_W + 1> primitive_table() {
    std::array<std::array<Rank64, MAX_W + 2>, MAX_W + 1> p{};
    p[0][0] = 1;
    for (int rem = 1; rem <= MAX_W; ++rem)
        for (int h = 0; h <= MAX_W; ++h)
            p[rem][h] = p[rem - 1][h + 1] + (h ? p[rem - 1][h - 1] : 0);
    return p;
}

Rank64 old_rank(
    MateID m, int len, int occupied,
    const std::array<std::array<Rank64, MAX_W + 2>, MAX_W + 1>& p
) {
    int h = 1;
    int seen = 0;
    Rank64 rank = 0;
    for (int pos = 0; pos < len; ++pos) {
        const int bit = len - 1 - pos;
        const MateValue c = MateValue((m >> (2 * bit)) & 3ULL);
        if (c == N) continue;
        const int rem = occupied - (++seen);
        if (c == L) {
            if (h > 0) rank += p[rem][h - 1];
            ++h;
        } else {
            --h;
        }
    }
    return rank;
}

Rank64 setbit_rank(
    MateID m, int occupied, std::uint32_t support,
    const std::array<std::array<Rank64, MAX_W + 2>, MAX_W + 1>& p
) {
    int h = 1;
    int seen = 0;
    Rank64 rank = 0;
    std::uint32_t mask = support;
    while (mask) {
        const int bit = 31 - __builtin_clz(mask);
        const MateValue c = MateValue((m >> (2 * bit)) & 3ULL);
        const int rem = occupied - (++seen);
        if (c == L) {
            if (h > 0) rank += p[rem][h - 1];
            ++h;
        } else {
            --h;
        }
        mask ^= std::uint32_t(1) << bit;
    }
    return rank;
}

bool check(MateID m, int W, std::uint32_t support, int occupied,
           const std::array<std::array<Rank64, MAX_W + 2>, MAX_W + 1>& p) {
    const Rank64 a = old_rank(m, W, occupied, p);
    const Rank64 b = setbit_rank(m, occupied, support, p);
    if (a == b) return true;
    std::cerr << "primitive setbit mismatch W=" << W
              << " mate=" << m << " support=" << support
              << " occupied=" << occupied << " old=" << a
              << " setbit=" << b << '\n';
    return false;
}

} // namespace

int main() {
    const auto primitive = primitive_table();
    std::uint64_t exhaustive_cases = 0;
    for (int W = 1; W <= 12; ++W) {
        std::uint64_t count = 1;
        for (int i = 0; i < W; ++i) count *= 3;
        for (std::uint64_t code = 0; code < count; ++code) {
            std::uint64_t x = code;
            MateID mate = 0;
            std::uint32_t support = 0;
            int occupied = 0;
            for (int bit = 0; bit < W; ++bit) {
                const auto v = static_cast<MateValue>(x % 3);
                x /= 3;
                mate |= MateID(v) << (2 * bit);
                if (v != N) {
                    support |= std::uint32_t(1) << bit;
                    ++occupied;
                }
            }
            if (!check(mate, W, support, occupied, primitive)) return 2;
            ++exhaustive_cases;
        }
    }

    std::uint64_t random_cases = 0;
    std::uint64_t s = 0x9e3779b97f4a7c15ULL;
    for (std::uint64_t tc = 0; tc < 1000000ULL; ++tc) {
        MateID mate = 0;
        std::uint32_t support = 0;
        int occupied = 0;
        for (int bit = 0; bit < MAX_W; ++bit) {
            s ^= s << 7;
            s ^= s >> 9;
            s ^= s << 8;
            const auto v = static_cast<MateValue>(s % 3);
            mate |= MateID(v) << (2 * bit);
            if (v != N) {
                support |= std::uint32_t(1) << bit;
                ++occupied;
            }
        }
        if (!check(mate, MAX_W, support, occupied, primitive)) return 3;
        ++random_cases;
    }

    std::cout << "gridfp-runtime-primitive-rank-setbits-proof OK"
              << " exhaustive_W_max=12"
              << " exhaustive_cases=" << exhaustive_cases
              << " random_W28=" << random_cases
              << " old_scans_per_rank_max=28"
              << " setbit_scans_per_rank=occupied"
              << " exact=1\n";
    return 0;
}
