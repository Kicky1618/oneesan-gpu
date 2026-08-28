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

std::uint64_t cache_word(const Key& k) {
    return std::uint64_t(k.mate) | (k.blocked ? (1ULL << 63) : 0ULL);
}

int cache_hash(const Key& k, int buckets) {
    std::uint64_t x = cache_word(k);
    x ^= x >> 7;
    x ^= x >> 14;
    return int(x & std::uint64_t(buckets - 1));
}

struct CacheStats {
    std::uint64_t calls = 0;
    std::uint64_t exact_comparisons = 0;
    std::uint64_t definite_misses = 0;
    std::uint64_t direct_hits = 0;
    std::uint64_t fallback_hits = 0;
    std::uint64_t collision_misses = 0;
};

int cache_find_recent(
    const std::vector<Key>& a,
    const std::array<int, 64>& latest,
    const Key& k,
    int buckets,
    CacheStats& st
) {
    ++st.calls;
    const int b = cache_hash(k, buckets);
    const int candidate = latest[std::size_t(b)];
    if (candidate < 0) {
        ++st.definite_misses;
        return -1;
    }
    if (candidate >= int(a.size())) fail("index-cache stale current-bucket index");
    ++st.exact_comparisons;
    if (same_key(a[std::size_t(candidate)], k)) {
        ++st.direct_hits;
        return candidate;
    }
    for (int i = int(a.size()) - 1; i >= 0; --i) {
        if (i == candidate) continue;
        ++st.exact_comparisons;
        if (same_key(a[std::size_t(i)], k)) {
            ++st.fallback_hits;
            return i;
        }
    }
    ++st.collision_misses;
    return -1;
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
    int buckets
) {
    CacheDiscoveryStats st;
    const std::vector<Key> seeds = runtime_like_seeds(labels, W, p, reverse);
    for (const Key& seed : seeds) {
        ++st.components;
        std::vector<Key> src{seed};
        std::vector<Key> dst;
        std::array<int, 64> src_latest{};
        std::array<int, 64> dst_latest{};
        src_latest.fill(-1);
        dst_latest.fill(-1);
        src_latest[std::size_t(cache_hash(seed, buckets))] = 0;
        std::size_t cursor = 0;
        while (cursor < src.size()) {
            const Key s = src[cursor++];
            const Vec edge = reduced_step_basis(s, W, p, reverse);
            for (const auto& [d, coef] : edge) {
                if (!coef) continue;
                ++st.edges;
                if (cache_find_recent(dst, dst_latest, d, buckets, st.destination_find) >= 0)
                    continue;
                const int di = int(dst.size());
                dst.push_back(d);
                dst_latest[std::size_t(cache_hash(d, buckets))] = di;
                const Vec pre = inverse_reduced(d, W, p, reverse);
                for (const auto& [x, a] : pre) {
                    if (!a) continue;
                    if (cache_find_recent(src, src_latest, x, buckets, st.source_find) >= 0)
                        continue;
                    const int si = int(src.size());
                    src.push_back(x);
                    src_latest[std::size_t(cache_hash(x, buckets))] = si;
                }
            }
        }
        if (src.size() != dst.size()) fail("index-cache unbalanced component");
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
    a.direct_hits += b.direct_hits;
    a.fallback_hits += b.fallback_hits;
    a.collision_misses += b.collision_misses;
}

void add_cache_discovery(CacheDiscoveryStats& a, const CacheDiscoveryStats& b) {
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

    std::vector<std::vector<MateID>> words(std::size_t(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 5; W <= maxW; ++W) {
        DiscoveryStats recent;
        std::array<CacheDiscoveryStats, 3> cached{};
        constexpr int buckets[3] = {16, 32, 64};
        for (bool reverse : {false, true}) {
            if (!reverse) {
                for (int p = W - 1; p >= 3; --p) {
                    add(recent, simulate_position(words[W - 1], W, p, false, true));
                    for (int i = 0; i < 3; ++i)
                        add_cache_discovery(cached[std::size_t(i)],
                            simulate_cached_position(words[W - 1], W, p, false, buckets[i]));
                }
            } else {
                for (int p = 1; p <= W - 3; ++p) {
                    add(recent, simulate_position(words[W - 1], W, p, true, true));
                    for (int i = 0; i < 3; ++i)
                        add_cache_discovery(cached[std::size_t(i)],
                            simulate_cached_position(words[W - 1], W, p, true, buckets[i]));
                }
            }
        }
        const std::uint64_t old_cmp = recent.destination_find.comparisons +
                                      recent.source_find.comparisons;
        for (int i = 0; i < 3; ++i) {
            const auto& c = cached[std::size_t(i)];
            if (recent.components != c.components ||
                recent.sources != c.sources ||
                recent.destinations != c.destinations ||
                recent.edges != c.edges)
                fail("index-cache changed traversal totals");
            const std::uint64_t new_cmp = c.destination_find.exact_comparisons +
                                          c.source_find.exact_comparisons;
            const std::uint64_t calls = c.destination_find.calls + c.source_find.calls;
            const std::uint64_t direct = c.destination_find.direct_hits + c.source_find.direct_hits;
            const std::uint64_t fallback = c.destination_find.fallback_hits + c.source_find.fallback_hits;
            const std::uint64_t definite = c.destination_find.definite_misses + c.source_find.definite_misses;
            std::cout << "W=" << W
                      << " buckets=" << buckets[i]
                      << " components=" << recent.components
                      << " find_calls=" << calls
                      << " recent_exact_comparisons=" << old_cmp
                      << " cached_exact_comparisons=" << new_cmp
                      << " comparison_ratio=" << std::fixed << std::setprecision(9)
                      << (old_cmp ? double(new_cmp) / double(old_cmp) : 1.0)
                      << " direct_hit_rate=" << (calls ? double(direct) / double(calls) : 0.0)
                      << " fallback_hit_rate=" << (calls ? double(fallback) / double(calls) : 0.0)
                      << " definite_miss_rate=" << (calls ? double(definite) / double(calls) : 0.0)
                      << " max_pairs=" << c.max_sources
                      << " traversal_exact=1\n";
        }
    }
    std::cout << "ALL_OK runtime_find_index_cache_model=1"
              << " hash=xor_shift_7_14 latest_index=1 stale_bytes_guarded_by_occupancy=1\n";
    return 0;
}
