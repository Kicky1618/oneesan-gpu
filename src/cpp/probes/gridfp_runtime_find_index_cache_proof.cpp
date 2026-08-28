#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

namespace {

constexpr std::uint8_t COLLIDED = 0x80u;
constexpr std::uint8_t VALUE_MASK = 0x7fu;

int bucket(std::uint64_t x, int buckets) {
    x ^= x >> 7;
    x ^= x >> 14;
    return int(x & std::uint64_t(buckets - 1));
}

int recent_find(const std::vector<std::uint64_t>& a, std::uint64_t k) {
    for (int i = int(a.size()) - 1; i >= 0; --i)
        if (a[std::size_t(i)] == k) return i;
    return -1;
}

int cached_find(
    const std::vector<std::uint64_t>& a,
    const std::array<std::uint8_t, 64>& cache,
    std::uint64_t occupancy,
    std::uint64_t k,
    int buckets,
    bool* collision_fallback = nullptr,
    bool* unique_bucket_miss = nullptr
) {
    const int b = bucket(k, buckets);
    const std::uint64_t bit = 1ULL << b;
    if ((occupancy & bit) == 0) return -1;
    const std::uint8_t packed = cache[std::size_t(b)];
    const int candidate = int(packed & VALUE_MASK) - 1;
    if (candidate >= 0 && candidate < int(a.size())) {
        if (a[std::size_t(candidate)] == k) return candidate;
        if ((packed & COLLIDED) == 0) {
            if (unique_bucket_miss) *unique_bucket_miss = true;
            return -1;
        }
        if (collision_fallback) *collision_fallback = true;
        for (int i = int(a.size()) - 1; i >= 0; --i) {
            if (i == candidate) continue;
            if (a[std::size_t(i)] == k) return i;
        }
        return -1;
    }
    return recent_find(a, k);
}

void record(
    std::array<std::uint8_t, 64>& cache,
    std::uint64_t& occupancy,
    std::uint64_t k,
    int index,
    int buckets
) {
    const int b = bucket(k, buckets);
    const std::uint64_t bit = 1ULL << b;
    const bool collided = (occupancy & bit) != 0;
    const std::uint8_t old = cache[std::size_t(b)];
    const std::uint8_t flags = collided
        ? std::uint8_t((old & COLLIDED) | COLLIDED)
        : 0u;
    cache[std::size_t(b)] =
        std::uint8_t(flags | std::uint8_t(index + 1));
    occupancy |= bit;
}

struct Stats {
    std::uint64_t exact_queries = 0;
    std::uint64_t stale_guard_queries = 0;
    std::uint64_t collision_fallback_queries = 0;
    std::uint64_t unique_bucket_miss_queries = 0;
};

Stats prove_bucket_count(int buckets, std::uint64_t seed) {
    constexpr int MAX_PAIRS = 20;
    constexpr std::uint64_t COMPONENTS = 100000;
    constexpr int PROBES_PER_COMPONENT = 32;

    std::array<std::uint8_t, 64> cache{};
    // Nonzero garbage, including collision bits, proves that occupancy alone
    // is enough to make stale bucket bytes harmless across components.
    for (int i = 0; i < 64; ++i)
        cache[std::size_t(i)] = std::uint8_t(0x80u | (1 + i % MAX_PAIRS));

    std::mt19937_64 rng(seed);
    Stats st{};
    for (std::uint64_t component = 0; component < COMPONENTS; ++component) {
        std::vector<std::uint64_t> keys;
        keys.reserve(MAX_PAIRS);
        std::uint64_t occupancy = 0;
        const int n = 1 + int(rng() % MAX_PAIRS);
        for (int i = 0; i < n; ++i) {
            std::uint64_t k;
            do { k = rng(); } while (recent_find(keys, k) >= 0);
            keys.push_back(k);
            record(cache, occupancy, k, i, buckets);
        }

        for (int q = 0; q < PROBES_PER_COMPONENT; ++q) {
            const std::uint64_t k = (q & 1)
                ? rng()
                : keys[std::size_t(rng() % keys.size())];
            const int want = recent_find(keys, k);
            bool collision_fallback = false;
            bool unique_bucket_miss = false;
            const int got = cached_find(
                keys, cache, occupancy, k, buckets,
                &collision_fallback, &unique_bucket_miss);
            ++st.exact_queries;
            const std::uint64_t bit = 1ULL << bucket(k, buckets);
            if ((occupancy & bit) == 0) ++st.stale_guard_queries;
            st.collision_fallback_queries += collision_fallback;
            st.unique_bucket_miss_queries += unique_bucket_miss;
            if (got != want) {
                std::cerr << "mismatch buckets=" << buckets
                          << " component=" << component
                          << " q=" << q << " got=" << got << " want=" << want << '\n';
                std::exit(2);
            }
        }
    }
    return st;
}

} // namespace

int main() {
    static_assert(sizeof(std::array<std::uint8_t, 64>) == 64);
    static_assert((COLLIDED & VALUE_MASK) == 0);
    static_assert(20 < VALUE_MASK);
    constexpr std::uint64_t COMPONENTS_PER_BUCKET = 100000;
    constexpr std::uint64_t QUERIES_PER_BUCKET = COMPONENTS_PER_BUCKET * 32;
    for (int buckets : {16, 32, 64}) {
        const Stats st = prove_bucket_count(
            buckets, 0x696e646578636163ULL ^ std::uint64_t(buckets));
        if (st.exact_queries != QUERIES_PER_BUCKET) return 3;
        if (!st.collision_fallback_queries || !st.unique_bucket_miss_queries) return 4;
        std::cout << "buckets=" << buckets
                  << " components=" << COMPONENTS_PER_BUCKET
                  << " exact_queries=" << st.exact_queries
                  << " stale_guard_queries=" << st.stale_guard_queries
                  << " collision_fallback_queries=" << st.collision_fallback_queries
                  << " unique_bucket_miss_queries=" << st.unique_bucket_miss_queries
                  << " bytes_per_set=" << buckets
                  << " bytes_per_subgroup=" << 2 * buckets
                  << " collision_state=packed_high_bit"
                  << " extra_collision_registers=0"
                  << " stale_bytes_clear_required=0 false_negative=0 exact=1\n";
    }
    std::cout << "gridfp-runtime-find-index-cache-proof OK"
              << " bucket_counts=16,32,64 max_pairs=20"
              << " total_exact_queries=" << 3 * QUERIES_PER_BUCKET
              << " collision_state=packed_high_bit extra_collision_registers=0"
              << " stale_bytes_clear_required=0 false_negative=0 exact=1\n";
    return 0;
}
