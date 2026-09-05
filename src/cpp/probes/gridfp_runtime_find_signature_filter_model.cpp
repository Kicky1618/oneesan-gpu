#pragma push_macro("main")
#undef main
#define main gridfp_runtime_discovery_find_work_probe_main_unused
#include "gridfp_runtime_discovery_find_work_probe.cpp"
#pragma pop_macro("main")

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

std::uint64_t signature_word(const Key& k) {
    return std::uint64_t(k.mate) | (k.blocked ? (1ULL << 63) : 0ULL);
}

std::uint64_t signature_bit(const Key& k) {
    std::uint64_t x = signature_word(k);
    x ^= x >> 7;
    x ^= x >> 14;
    return 1ULL << (x & 63ULL);
}

struct FilterStats {
    std::uint64_t calls = 0;
    std::uint64_t exact_comparisons = 0;
    std::uint64_t definite_misses = 0;
    std::uint64_t false_positive_misses = 0;
    std::uint64_t hits = 0;
};

int filtered_find_recent(
    const std::vector<Key>& a,
    std::uint64_t filter,
    const Key& k,
    FilterStats& st
) {
    ++st.calls;
    if ((filter & signature_bit(k)) == 0) {
        ++st.definite_misses;
        return -1;
    }
    for (int i = int(a.size()) - 1; i >= 0; --i) {
        ++st.exact_comparisons;
        if (same_key(a[std::size_t(i)], k)) {
            ++st.hits;
            return i;
        }
    }
    ++st.false_positive_misses;
    return -1;
}

struct FilterDiscoveryStats {
    std::uint64_t components = 0;
    std::uint64_t sources = 0;
    std::uint64_t destinations = 0;
    std::uint64_t edges = 0;
    FilterStats destination_find;
    FilterStats source_find;
    std::uint64_t max_sources = 0;
};

FilterDiscoveryStats simulate_filtered_position(
    const std::vector<MateID>& labels,
    int W,
    int p,
    bool reverse
) {
    FilterDiscoveryStats st;
    const std::vector<Key> seeds = runtime_like_seeds(labels, W, p, reverse);
    for (const Key& seed : seeds) {
        ++st.components;
        std::vector<Key> src{seed};
        std::vector<Key> dst;
        std::uint64_t src_filter = signature_bit(seed);
        std::uint64_t dst_filter = 0;
        std::size_t cursor = 0;
        while (cursor < src.size()) {
            const Key s = src[cursor++];
            const Vec edge = reduced_step_basis(s, W, p, reverse);
            for (const auto& [d, coef] : edge) {
                if (!coef) continue;
                ++st.edges;
                if (filtered_find_recent(dst, dst_filter, d, st.destination_find) >= 0)
                    continue;
                dst.push_back(d);
                dst_filter |= signature_bit(d);
                const Vec pre = inverse_reduced(d, W, p, reverse);
                for (const auto& [x, a] : pre) {
                    if (!a) continue;
                    if (filtered_find_recent(src, src_filter, x, st.source_find) >= 0)
                        continue;
                    src.push_back(x);
                    src_filter |= signature_bit(x);
                }
            }
        }
        if (src.size() != dst.size()) fail("signature-filter unbalanced component");
        st.sources += src.size();
        st.destinations += dst.size();
        st.max_sources = std::max<std::uint64_t>(st.max_sources, src.size());
    }
    return st;
}

void add_filter(FilterStats& a, const FilterStats& b) {
    a.calls += b.calls;
    a.exact_comparisons += b.exact_comparisons;
    a.definite_misses += b.definite_misses;
    a.false_positive_misses += b.false_positive_misses;
    a.hits += b.hits;
}

void add_filter_discovery(FilterDiscoveryStats& a, const FilterDiscoveryStats& b) {
    a.components += b.components;
    a.sources += b.sources;
    a.destinations += b.destinations;
    a.edges += b.edges;
    add_filter(a.destination_find, b.destination_find);
    add_filter(a.source_find, b.source_find);
    a.max_sources = std::max(a.max_sources, b.max_sources);
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 11;
    if (maxW < 5 || maxW > 12) return 2;

    std::vector<std::vector<MateID>> words(std::size_t(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 5; W <= maxW; ++W) {
        DiscoveryStats recent;
        FilterDiscoveryStats filtered;
        for (bool reverse : {false, true}) {
            if (!reverse) {
                for (int p = W - 1; p >= 3; --p) {
                    add(recent, simulate_position(words[W - 1], W, p, false, true));
                    add_filter_discovery(
                        filtered, simulate_filtered_position(words[W - 1], W, p, false));
                }
            } else {
                for (int p = 1; p <= W - 3; ++p) {
                    add(recent, simulate_position(words[W - 1], W, p, true, true));
                    add_filter_discovery(
                        filtered, simulate_filtered_position(words[W - 1], W, p, true));
                }
            }
        }

        if (recent.components != filtered.components ||
            recent.sources != filtered.sources ||
            recent.destinations != filtered.destinations ||
            recent.edges != filtered.edges)
            fail("signature-filter changed traversal totals");

        const std::uint64_t old_cmp = recent.destination_find.comparisons +
                                      recent.source_find.comparisons;
        const std::uint64_t new_cmp = filtered.destination_find.exact_comparisons +
                                      filtered.source_find.exact_comparisons;
        const std::uint64_t calls = filtered.destination_find.calls +
                                    filtered.source_find.calls;
        const std::uint64_t definite = filtered.destination_find.definite_misses +
                                       filtered.source_find.definite_misses;
        const std::uint64_t fp = filtered.destination_find.false_positive_misses +
                                 filtered.source_find.false_positive_misses;
        std::cout << "W=" << W
                  << " components=" << recent.components
                  << " find_calls=" << calls
                  << " recent_exact_comparisons=" << old_cmp
                  << " filtered_exact_comparisons=" << new_cmp
                  << " comparison_ratio=" << std::fixed << std::setprecision(9)
                  << (old_cmp ? double(new_cmp) / double(old_cmp) : 1.0)
                  << " definite_miss_rate="
                  << (calls ? double(definite) / double(calls) : 0.0)
                  << " false_positive_miss_rate="
                  << (calls ? double(fp) / double(calls) : 0.0)
                  << " max_pairs=" << filtered.max_sources
                  << " traversal_exact=1\n";
    }
    std::cout << "ALL_OK runtime_find_signature_filter_model=1"
              << " hash=xor_shift_7_14 bits=64 false_negative=0\n";
    return 0;
}
