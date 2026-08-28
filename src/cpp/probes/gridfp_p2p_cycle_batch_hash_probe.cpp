#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>

namespace {

std::uint32_t mask_bits(int len) {
    return len == 32 ? ~std::uint32_t(0)
                     : (std::uint32_t(1) << len) - 1u;
}

std::uint32_t rot(std::uint32_t x, int len, int shift) {
    const std::uint32_t mask = mask_bits(len);
    shift %= len;
    if (shift < 0) shift += len;
    x &= mask;
    if (!shift) return x;
    return ((x << shift) | (x >> (len - shift))) & mask;
}

unsigned pop(std::uint32_t x) {
    return static_cast<unsigned>(__builtin_popcount(x));
}

std::uint32_t mix32(std::uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

std::uint32_t main_hash(std::uint32_t support, int W) {
    std::uint32_t h = pop(support) * 0x9e3779b1u;
    h ^= pop(support & rot(support, W, 1)) * 0x85ebca6bu;
    h ^= pop(support & rot(support, W, 3)) * 0xc2b2ae35u;
    h ^= pop(support & rot(support, W, 5)) * 0x27d4eb2fu;
    h ^= pop(support & rot(support, W, 7)) * 0x165667b1u;
    return mix32(h);
}

std::uint32_t blocked_hash_compact(std::uint32_t compact, int K) {
    const std::uint32_t half_mask = mask_bits(K);
    const std::uint32_t a = compact & half_mask;
    const std::uint32_t b = (compact >> K) & half_mask;
    const std::uint32_t lo = std::min(a, b);
    const std::uint32_t hi = std::max(a, b);
    std::uint32_t h = lo * 0x9e3779b1u;
    h ^= hi * 0x85ebca6bu;
    h ^= pop(compact) * 0xc2b2ae35u;
    return mix32(h);
}

} // namespace

int main(int argc, char** argv) {
    const int max_w = argc > 1 ? std::atoi(argv[1]) : 18;
    if (max_w < 8 || max_w > 24) return 2;

    std::uint64_t main_cases = 0;
    std::uint64_t blocked_cases = 0;
    for (int W = 8; W <= max_w; W += 2) {
        const int K = (W - 2) / 2;
        const std::uint32_t main_count = std::uint32_t(1) << W;
        for (std::uint32_t x = 0; x < main_count; ++x) {
            const std::uint32_t want = main_hash(x, W);
            std::uint32_t cur = x;
            for (int hop = 0; hop < W; ++hop) {
                if (main_hash(cur, W) != want) {
                    std::cerr << "main hash changed W=" << W
                              << " x=" << x << " hop=" << hop << '\n';
                    return 3;
                }
                cur = rot(cur, W, K);
                ++main_cases;
            }
        }

        const int compact_len = 2 * K;
        const std::uint32_t blocked_count = std::uint32_t(1) << compact_len;
        for (std::uint32_t x = 0; x < blocked_count; ++x) {
            const std::uint32_t want = blocked_hash_compact(x, K);
            const std::uint32_t swapped = rot(x, compact_len, K);
            if (blocked_hash_compact(swapped, K) != want) {
                std::cerr << "blocked hash changed W=" << W
                          << " compact=" << x << '\n';
                return 4;
            }
            blocked_cases += 2;
        }
    }

    // Production-specific algebraic shape: W=28, K=S=13.  Test a dense
    // deterministic sample without enumerating 2^28 masks on the CPU.
    std::uint32_t x = 0x13579bdu;
    for (std::uint64_t i = 0; i < 4'000'000ULL; ++i) {
        x = x * 1664525u + 1013904223u;
        x &= mask_bits(28);
        const std::uint32_t h = main_hash(x, 28);
        std::uint32_t cur = x;
        for (int hop = 0; hop < 28; ++hop) {
            if (main_hash(cur, 28) != h) return 5;
            cur = rot(cur, 28, 13);
        }
        const std::uint32_t compact = x & mask_bits(26);
        if (blocked_hash_compact(compact, 13) !=
            blocked_hash_compact(rot(compact, 26, 13), 13)) return 6;
    }

    std::cout << "ALL_OK"
              << " cycle_batch_hash=1"
              << " main_rotation_invariant=1"
              << " blocked_half_swap_invariant=1"
              << " exhaustive_max_W=" << max_w
              << " main_checks=" << main_cases
              << " blocked_checks=" << blocked_cases
              << " production_samples=4000000"
              << " supported_batches=2,4,8,16,32\n";
    return 0;
}
