#pragma push_macro("main")
#undef main
#define main gridfp_runtime_discovery_find_work_probe_main_unused
#include "gridfp_runtime_discovery_find_work_probe.cpp"
#pragma pop_macro("main")

#include <array>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

constexpr std::uint8_t CACHE_OVERFLOW = 0x80u;
constexpr std::uint8_t CACHE_VALUE_MASK = 0x7fu;
constexpr int MAX_WAYS = 4;
using Cache = std::array<std::array<std::uint8_t, MAX_WAYS>, 64>;

struct CacheConfig {
    int storage_bytes = 64;
    int ways = 1;
    int hash_buckets() const { return storage_bytes / ways; }
};

std::uint64_t cache_word(const Key& k) {
    return std::uint64_t(k.mate) | (k.blocked ? (1ULL << 63) : 0ULL);
}

int cache_hash(const Key& k, int hash_buckets) {
    std::uint64_t x = cache_word(k);
    x ^= x >> 7;
    x ^= x >> 14;
    return int(x & std::uint64_t(hash_buckets - 1));
}

struct CacheStats {
    std::uint64_t calls = 0;
    std::uint64_t exact_comparisons = 0;
    std::uint64_t definite_misses = 0;
    std::uint64_t retained_hits = 0;
    std::uint64_t retained_bucket_misses = 0;
    std::uint64_t overflow_fallback_calls = 0;
    std::uint64_t fallback_hits = 0;
    std::uint64_t overflow_misses = 0;
};

bool cache_index_retained(
    const Cache& cache, int bucket, int ways, int index
) {
    for (int way = 0; way < ways; ++way) {
        const int v = int(cache[std::size_t(bucket)][std::size_t(way)] &
                          CACHE_VALUE_MASK);
        if (!v) break;
        if (v - 1 == index) return true;
    }
    return false;
}

int cache_find_recent(
    const std::vector<Key>& a,
    const Cache& cache,
    std::uint64_t occupancy,
    const Key& k,
    CacheConfig cfg,
    CacheStats& st
) {
    ++st.calls;
    const int buckets = cfg.hash_buckets();
    const int b = cache_hash(k, buckets);
    const std::uint64_t bit = 1ULL << b;
    if ((occupancy & bit) == 0) {
        ++st.definite_misses;
        return -1;
    }

    const bool overflow =
        (cache[std::size_t(b)][0] & CACHE_OVERFLOW) != 0;
    for (int way = 0; way < cfg.ways; ++way) {
        const int v = int(cache[std::size_t(b)][std::size_t(way)] &
                          CACHE_VALUE_MASK);
        if (!v) break;
        const int candidate = v - 1;
        if (candidate < 0 || candidate >= int(a.size()))
            fail("index-cache stale current-bucket index");
        ++st.exact_comparisons;
        if (same_key(a[std::size_t(candidate)], k)) {
            ++st.retained_hits;
            return candidate;
        }
    }

    if (!overflow) {
        ++st.retained_bucket_misses;
        return -1;
    }

    ++st.overflow_fallback_calls;
    for (int i = int(a.size()) - 1; i >= 0; --i) {
        if (cache_index_retained(cache, b, cfg.ways, i)) continue;
        ++st.exact_comparisons;
        if (same_key(a[std::size_t(i)], k)) {
            ++st.fallback_hits;
            return i;
        }
    }
    ++st.overflow_misses;
    return -1;
}

void cache_record(
    Cache& cache,
    std::uint64_t& occupancy,
    const Key& k,
    int index,
    CacheConfig cfg
) {
    const int b = cache_hash(k, cfg.hash_buckets());
    const std::uint64_t bit = 1ULL << b;
    auto& row = cache[std::size_t(b)];
    if ((occupancy & bit) == 0) {
        row[0] = std::uint8_t(index + 1);
        for (int way = 1; way < cfg.ways; ++way)
            row[std::size_t(way)] = 0;
        occupancy |= bit;
        return;
    }

    bool overflow = (row[0] & CACHE_OVERFLOW) != 0;
    overflow |= (row[std::size_t(cfg.ways - 1)] & CACHE_VALUE_MASK) != 0;
    for (int way = cfg.ways - 1; way > 0; --way) {
        row[std::size_t(way)] =
            row[std::size_t(way - 1)] & CACHE_VALUE_MASK;
    }
    row[0] = std::uint8_t(
        std::uint8_t(index + 1) |
        (overflow ? CACHE_OVERFLOW : 0u));
}

struct CacheDiscoveryStats {
    std::uint64_t components = 0;
    std::uint64_t sources = 0;
    std::uint64_t destinations = 0;
    std::uint64_t edges = 0;
    CacheStats destination_find;
    CacheStats source_find;
    std::uint64_t max_sources = 0;
};

CacheDiscoveryStats simulate_cached_position(
    const std::vector<MateID>& labels,
    int W,
    int p,
    bool reverse,
    CacheConfig cfg
) {
    CacheDiscoveryStats st;
    const std::vector<Key> seeds = runtime_like_seeds(labels, W, p, reverse);
    for (const Key& seed : seeds) {
        ++st.components;
        std::vector<Key> src{seed};
        std::vector<Key> dst;
        Cache src_cache{};
        Cache dst_cache{};
        std::uint64_t src_occupancy = 0;
        std::uint64_t dst_occupancy = 0;
        cache_record(src_cache, src_occupancy, seed, 0, cfg);
        std::size_t cursor = 0;
        while (cursor < src.size()) {
            const Key s = src[cursor++];
            const Vec edge = reduced_step_basis(s, W, p, reverse);
            for (const auto& [d, coef] : edge) {
                if (!coef) continue;
                ++st.edges;
                if (cache_find_recent(
                        dst, dst_cache, dst_occupancy, d, cfg,
                        st.destination_find) >= 0)
                    continue;
                const int di = int(dst.size());
                dst.push_back(d);
                cache_record(dst_cache, dst_occupancy, d, di, cfg);
                const Vec pre = inverse_reduced(d, W, p, reverse);
                for (const auto& [x, a] : pre) {
                    if (!a) continue;
                    if (cache_find_recent(
                            src, src_cache, src_occupancy, x, cfg,
                            st.source_find) >= 0)
                        continue;
                    const int si = int(src.size());
                    src.push_back(x);
                    cache_record(src_cache, src_occupancy, x, si, cfg);
                }
            }
        }
        if (src.size() != dst.size())
            fail("index-cache unbalanced component");
        st.sources += src.size();
        st.destinations += dst.size();
        st.max_sources = std::max<std::uint64_t>(st.max_sources, src.size());
    }
    return st;
}

void add_cache(CacheStats& a, const CacheStats& b) {
    a.calls += b.calls;
    a.exact_comparisons += b.exact_comparisons;
    a.definite_misses += b.definite_misses;
    a.retained_hits += b.retained_hits;
    a.retained_bucket_misses += b.retained_bucket_misses;
    a.overflow_fallback_calls += b.overflow_fallback_calls;
    a.fallback_hits += b.fallback_hits;
    a.overflow_misses += b.overflow_misses;
}

void add_cache_discovery(
    CacheDiscoveryStats& a, const CacheDiscoveryStats& b
) {
    a.components += b.components;
    a.sources += b.sources;
    a.destinations += b.destinations;
    a.edges += b.edges;
    add_cache(a.destination_find, b.destination_find);
    add_cache(a.source_find, b.source_find);
    a.max_sources = std::max(a.max_sources, b.max_sources);
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 10;
    if (maxW < 5 || maxW > 12) return 2;

    constexpr CacheConfig configs[] = {
        {16,1}, {32,1}, {64,1}, {32,2}, {64,2}, {64,4}
    };
    std::vector<std::vector<MateID>> words(std::size_t(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 5; W <= maxW; ++W) {
        DiscoveryStats recent;
        std::array<CacheDiscoveryStats, 6> cached{};
        for (bool reverse : {false, true}) {
            if (!reverse) {
                for (int p = W - 1; p >= 3; --p) {
                    add(recent,
                        simulate_position(words[W - 1], W, p, false, true));
                    for (int i = 0; i < 6; ++i)
                        add_cache_discovery(
                            cached[std::size_t(i)],
                            simulate_cached_position(
                                words[W - 1], W, p, false, configs[i]));
                }
            } else {
                for (int p = 1; p <= W - 3; ++p) {
                    add(recent,
                        simulate_position(words[W - 1], W, p, true, true));
                    for (int i = 0; i < 6; ++i)
                        add_cache_discovery(
                            cached[std::size_t(i)],
                            simulate_cached_position(
                                words[W - 1], W, p, true, configs[i]));
                }
            }
        }

        const std::uint64_t old_cmp =
            recent.destination_find.comparisons + recent.source_find.comparisons;
        for (int i = 0; i < 6; ++i) {
            const CacheConfig cfg = configs[i];
            const auto& c = cached[std::size_t(i)];
            if (recent.components != c.components ||
                recent.sources != c.sources ||
                recent.destinations != c.destinations ||
                recent.edges != c.edges)
                fail("index-cache changed traversal totals");
            const std::uint64_t new_cmp =
                c.destination_find.exact_comparisons +
                c.source_find.exact_comparisons;
            const std::uint64_t calls =
                c.destination_find.calls + c.source_find.calls;
            const std::uint64_t retained_hits =
                c.destination_find.retained_hits + c.source_find.retained_hits;
            const std::uint64_t fallback_hits =
                c.destination_find.fallback_hits + c.source_find.fallback_hits;
            const std::uint64_t definite =
                c.destination_find.definite_misses + c.source_find.definite_misses;
            const std::uint64_t retained_miss =
                c.destination_find.retained_bucket_misses +
                c.source_find.retained_bucket_misses;
            const std::uint64_t fallback_calls =
                c.destination_find.overflow_fallback_calls +
                c.source_find.overflow_fallback_calls;
            std::cout << "W=" << W
                      << " storage_bytes=" << cfg.storage_bytes
                      << " ways=" << cfg.ways
                      << " hash_buckets=" << cfg.hash_buckets()
                      << " components=" << recent.components
                      << " find_calls=" << calls
                      << " recent_exact_comparisons=" << old_cmp
                      << " cached_exact_comparisons=" << new_cmp
                      << " comparison_ratio=" << std::fixed << std::setprecision(9)
                      << (old_cmp ? double(new_cmp) / double(old_cmp) : 1.0)
                      << " retained_hit_rate="
                      << (calls ? double(retained_hits) / double(calls) : 0.0)
                      << " fallback_hit_rate="
                      << (calls ? double(fallback_hits) / double(calls) : 0.0)
                      << " definite_miss_rate="
                      << (calls ? double(definite) / double(calls) : 0.0)
                      << " retained_bucket_miss_rate="
                      << (calls ? double(retained_miss) / double(calls) : 0.0)
                      << " overflow_fallback_rate="
                      << (calls ? double(fallback_calls) / double(calls) : 0.0)
                      << " max_pairs=" << c.max_sources
                      << " traversal_exact=1\n";
        }
    }
    std::cout << "ALL_OK runtime_find_index_cache_model=1"
              << " hash=xor_shift_7_14 set_associative=1"
              << " overflow_state=packed_high_bit"
              << " extra_overflow_registers=0"
              << " stale_bytes_guarded_by_occupancy=1\n";
    return 0;
}
