#include <array>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

namespace {

int bucket(std::uint64_t x) {
    x ^= x >> 7;
    x ^= x >> 14;
    return int(x & 63ULL);
}

int recent_find(const std::vector<std::uint64_t>& a, std::uint64_t k) {
    for (int i = int(a.size()) - 1; i >= 0; --i)
        if (a[std::size_t(i)] == k) return i;
    return -1;
}

int cached_find(
    const std::vector<std::uint64_t>& a,
    const std::array<std::uint8_t, 64>& latest_plus_one,
    std::uint64_t occupancy,
    std::uint64_t k
) {
    const int b = bucket(k);
    const std::uint64_t bit = 1ULL << b;
    if ((occupancy & bit) == 0) return -1;
    const int candidate = int(latest_plus_one[std::size_t(b)]) - 1;
    if (candidate >= 0 && candidate < int(a.size())) {
        if (a[std::size_t(candidate)] == k) return candidate;
        for (int i = int(a.size()) - 1; i >= 0; --i) {
            if (i == candidate) continue;
            if (a[std::size_t(i)] == k) return i;
        }
        return -1;
    }
    return recent_find(a, k);
}

void record(
    std::array<std::uint8_t, 64>& latest_plus_one,
    std::uint64_t& occupancy,
    std::uint64_t k,
    int index
) {
    const int b = bucket(k);
    latest_plus_one[std::size_t(b)] = std::uint8_t(index + 1);
    occupancy |= 1ULL << b;
}

} // namespace

int main() {
    static_assert(sizeof(std::array<std::uint8_t, 64>) == 64);
    constexpr int MAX_PAIRS = 20;
    constexpr std::uint64_t COMPONENTS = 200000;
    constexpr int PROBES_PER_COMPONENT = 32;

    std::array<std::uint8_t, 64> cache{};
    // Fill with nonzero garbage once. Every subsequent component deliberately
    // reuses these stale bytes and resets only the occupancy word.
    for (int i = 0; i < 64; ++i) cache[std::size_t(i)] = std::uint8_t(1 + i % MAX_PAIRS);

    std::mt19937_64 rng(0x696e646578636163ULL);
    std::uint64_t exact_queries = 0;
    std::uint64_t stale_guard_queries = 0;
    std::uint64_t collision_queries = 0;

    for (std::uint64_t component = 0; component < COMPONENTS; ++component) {
        std::vector<std::uint64_t> keys;
        keys.reserve(MAX_PAIRS);
        std::uint64_t occupancy = 0;
        const int n = 1 + int(rng() % MAX_PAIRS);
        for (int i = 0; i < n; ++i) {
            std::uint64_t k;
            do {
                k = rng();
            } while (recent_find(keys, k) >= 0);
            keys.push_back(k);
            record(cache, occupancy, k, i);
        }

        for (int q = 0; q < PROBES_PER_COMPONENT; ++q) {
            std::uint64_t k;
            if ((q & 1) == 0) {
                k = keys[std::size_t(rng() % keys.size())];
            } else {
                k = rng();
            }
            const int want = recent_find(keys, k);
            const int got = cached_find(keys, cache, occupancy, k);
            ++exact_queries;
            if ((occupancy & (1ULL << bucket(k))) == 0) ++stale_guard_queries;
            if ((occupancy & (1ULL << bucket(k))) != 0) {
                const int c = int(cache[std::size_t(bucket(k))]) - 1;
                if (c >= 0 && c < int(keys.size()) && keys[std::size_t(c)] != k)
                    ++collision_queries;
            }
            if (got != want) {
                std::cerr << "mismatch component=" << component
                          << " q=" << q << " got=" << got << " want=" << want << '\n';
                return 2;
            }
        }
    }

    std::cout << "gridfp-runtime-find-index-cache-proof OK"
              << " components=" << COMPONENTS
              << " exact_queries=" << exact_queries
              << " stale_guard_queries=" << stale_guard_queries
              << " collision_queries=" << collision_queries
              << " max_pairs=" << MAX_PAIRS
              << " buckets=64 bytes_per_set=64 bytes_per_subgroup=128"
              << " stale_bytes_clear_required=0 false_negative=0 exact=1\n";
    return 0;
}
