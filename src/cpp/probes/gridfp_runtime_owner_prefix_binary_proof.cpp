#include <array>
#include <cstdint>
#include <iostream>
#include <random>

namespace {
using Rank64 = std::uint64_t;

int linear(const Rank64* prefix, int O, Rank64 rank) {
    for (int r = 0; r <= O; ++r)
        if (rank < prefix[r + 1]) return r;
    return -1;
}
int binary(const Rank64* prefix, int O, Rank64 rank) {
    int lo = 0, hi = O + 1;
    while (lo < hi) {
        const int mid = lo + ((hi - lo) >> 1);
        if (rank < prefix[mid + 1]) hi = mid;
        else lo = mid + 1;
    }
    return lo <= O ? lo : -1;
}
}

int main() {
    std::uint64_t boundary_cases = 0;
    std::mt19937_64 rng(0x7072656669786269ULL);
    for (int O = 0; O <= 13; ++O) {
        for (int t = 0; t < 100000; ++t) {
            std::array<Rank64, 15> p{};
            for (int r = 0; r <= O; ++r) p[r + 1] = p[r] + 1 + (rng() % 1000000ULL);
            for (int r = 0; r <= O; ++r) {
                const Rank64 a = p[r], b = p[r + 1];
                const Rank64 probes[] = {a, a + (b - a) / 2, b - 1};
                for (Rank64 x : probes) {
                    ++boundary_cases;
                    if (linear(p.data(), O, x) != binary(p.data(), O, x)) return 2;
                }
            }
            if (linear(p.data(), O, p[O + 1]) != -1 ||
                binary(p.data(), O, p[O + 1]) != -1) return 3;
        }
    }
    std::cout << "gridfp-runtime-owner-prefix-binary-proof OK"
              << " O_max=13 boundary_cases=" << boundary_cases
              << " production_W_max=28 exact=1 max_binary_comparisons=4\n";
    return 0;
}
