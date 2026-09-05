#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

namespace {

constexpr std::uint8_t OVERFLOW = 0x80u;
constexpr std::uint8_t VALUE_MASK = 0x7fu;
constexpr int MAX_WAYS = 4;
using Cache = std::array<std::array<std::uint8_t, MAX_WAYS>, 64>;

int bucket(std::uint64_t x, int hash_buckets) {
    x ^= x >> 7;
    x ^= x >> 14;
    return int(x & std::uint64_t(hash_buckets - 1));
}

int recent_find(const std::vector<std::uint64_t>& a, std::uint64_t k) {
    for (int i = int(a.size()) - 1; i >= 0; --i)
        if (a[std::size_t(i)] == k) return i;
    return -1;
}

bool cached_index(const Cache& cache, int b, int ways, int index) {
    for (int way = 0; way < ways; ++way) {
        const int v = int(cache[std::size_t(b)][std::size_t(way)] & VALUE_MASK);
        if (!v) break;
        if (v - 1 == index) return true;
    }
    return false;
}

int cached_find(
    const std::vector<std::uint64_t>& a,
    const Cache& cache,
    std::uint64_t occupancy,
    std::uint64_t k,
    int hash_buckets,
    int ways,
    bool* overflow_fallback = nullptr,
    bool* retained_bucket_miss = nullptr
) {
    const int b = bucket(k, hash_buckets);
    const std::uint64_t bit = 1ULL << b;
    if ((occupancy & bit) == 0) return -1;
    const bool overflow =
        (cache[std::size_t(b)][0] & OVERFLOW) != 0;
    for (int way = 0; way < ways; ++way) {
        const int v = int(cache[std::size_t(b)][std::size_t(way)] & VALUE_MASK);
        if (!v) break;
        const int candidate = v - 1;
        if (candidate < 0 || candidate >= int(a.size())) return recent_find(a, k);
        if (a[std::size_t(candidate)] == k) return candidate;
    }
    if (!overflow) {
        if (retained_bucket_miss) *retained_bucket_miss = true;
        return -1;
    }
    if (overflow_fallback) *overflow_fallback = true;
    for (int i = int(a.size()) - 1; i >= 0; --i) {
        if (cached_index(cache, b, ways, i)) continue;
        if (a[std::size_t(i)] == k) return i;
    }
    return -1;
}

void record(
    Cache& cache,
    std::uint64_t& occupancy,
    std::uint64_t k,
    int index,
    int hash_buckets,
    int ways
) {
    const int b = bucket(k, hash_buckets);
    const std::uint64_t bit = 1ULL << b;
    auto& row = cache[std::size_t(b)];
    if ((occupancy & bit) == 0) {
        row[0] = std::uint8_t(index + 1);
        for (int way = 1; way < ways; ++way) row[std::size_t(way)] = 0;
        occupancy |= bit;
        return;
    }
    bool overflow = (row[0] & OVERFLOW) != 0;
    overflow |= (row[std::size_t(ways - 1)] & VALUE_MASK) != 0;
    for (int way = ways - 1; way > 0; --way)
        row[std::size_t(way)] = row[std::size_t(way - 1)] & VALUE_MASK;
    row[0] = std::uint8_t(
        std::uint8_t(index + 1) | (overflow ? OVERFLOW : 0u));
}

struct Stats {
    std::uint64_t exact_queries = 0;
    std::uint64_t stale_guard_queries = 0;
    std::uint64_t overflow_fallback_queries = 0;
    std::uint64_t retained_bucket_miss_queries = 0;
};

Stats prove_config(int storage_bytes, int ways, std::uint64_t seed) {
    constexpr int MAX_PAIRS = 20;
    constexpr std::uint64_t COMPONENTS = 50000;
    constexpr int PROBES_PER_COMPONENT = 32;
    const int hash_buckets = storage_bytes / ways;

    Cache cache{};
    for (auto& row : cache)
        for (auto& v : row) v = 0x91u;

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
            record(cache, occupancy, k, i, hash_buckets, ways);
        }

        for (int q = 0; q < PROBES_PER_COMPONENT; ++q) {
            const std::uint64_t k = (q & 1)
                ? rng()
                : keys[std::size_t(rng() % keys.size())];
            const int want = recent_find(keys, k);
            bool overflow_fallback = false;
            bool retained_bucket_miss = false;
            const int got = cached_find(
                keys, cache, occupancy, k, hash_buckets, ways,
                &overflow_fallback, &retained_bucket_miss);
            ++st.exact_queries;
            const std::uint64_t bit = 1ULL << bucket(k, hash_buckets);
            if ((occupancy & bit) == 0) ++st.stale_guard_queries;
            st.overflow_fallback_queries += overflow_fallback;
            st.retained_bucket_miss_queries += retained_bucket_miss;
            if (got != want) {
                std::cerr << "mismatch storage=" << storage_bytes
                          << " ways=" << ways
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
    static_assert((OVERFLOW & VALUE_MASK) == 0);
    static_assert(20 < VALUE_MASK);
    struct Config { int storage; int ways; };
    constexpr Config configs[] = {
        {16,1}, {32,1}, {64,1}, {32,2}, {64,2}, {64,4}
    };
    constexpr std::uint64_t COMPONENTS_PER_CONFIG = 50000;
    constexpr std::uint64_t QUERIES_PER_CONFIG = COMPONENTS_PER_CONFIG * 32;
    std::uint64_t total_queries = 0;
    for (const Config cfg : configs) {
        const int hash_buckets = cfg.storage / cfg.ways;
        const Stats st = prove_config(
            cfg.storage, cfg.ways,
            0x696e646578636163ULL ^ std::uint64_t(cfg.storage << 4 | cfg.ways));
        if (st.exact_queries != QUERIES_PER_CONFIG) return 3;
        if (!st.retained_bucket_miss_queries) return 4;
        total_queries += st.exact_queries;
        std::cout << "storage_bytes=" << cfg.storage
                  << " ways=" << cfg.ways
                  << " hash_buckets=" << hash_buckets
                  << " components=" << COMPONENTS_PER_CONFIG
                  << " exact_queries=" << st.exact_queries
                  << " stale_guard_queries=" << st.stale_guard_queries
                  << " overflow_fallback_queries=" << st.overflow_fallback_queries
                  << " retained_bucket_miss_queries=" << st.retained_bucket_miss_queries
                  << " bytes_per_subgroup=" << 2 * cfg.storage
                  << " overflow_state=packed_high_bit"
                  << " extra_overflow_registers=0"
                  << " stale_bytes_clear_required=0 false_negative=0 exact=1\n";
    }
    std::cout << "gridfp-runtime-find-index-cache-proof OK"
              << " configs=6 max_pairs=20 total_exact_queries=" << total_queries
              << " set_associative=1 overflow_state=packed_high_bit"
              << " extra_overflow_registers=0"
              << " stale_bytes_clear_required=0 false_negative=0 exact=1\n";
    return 0;
}
