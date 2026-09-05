#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_inverse_probe_main_unused
#include "gridfp_reduced_production_inverse_probe.cpp"
#pragma pop_macro("main")

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

bool same_key(const Key& a, const Key& b) {
    return a.blocked == b.blocked && a.mate == b.mate;
}

struct FindStats {
    std::uint64_t calls = 0;
    std::uint64_t comparisons = 0;
    std::uint64_t hits = 0;
    std::uint64_t misses = 0;
    std::uint64_t hit_comparisons = 0;
    std::uint64_t miss_comparisons = 0;
    std::uint64_t max_n = 0;
};

int find_counted(const std::vector<Key>& a, const Key& k, bool recent_first, FindStats& st) {
    ++st.calls;
    st.max_n = std::max<std::uint64_t>(st.max_n, a.size());
    std::uint64_t local = 0;
    if (recent_first) {
        for (int i = int(a.size()) - 1; i >= 0; --i) {
            ++local;
            if (same_key(a[std::size_t(i)], k)) {
                ++st.hits;
                st.comparisons += local;
                st.hit_comparisons += local;
                return i;
            }
        }
    } else {
        for (std::size_t i = 0; i < a.size(); ++i) {
            ++local;
            if (same_key(a[i], k)) {
                ++st.hits;
                st.comparisons += local;
                st.hit_comparisons += local;
                return int(i);
            }
        }
    }
    ++st.misses;
    st.comparisons += local;
    st.miss_comparisons += local;
    return -1;
}

std::vector<Key> runtime_like_seeds(
    const std::vector<MateID>& labels, int W, int p, bool reverse
) {
    std::vector<Key> out;
    for (MateID v : labels) {
        const int other = reverse ? p : p - 2;
        if (mget(v, p - 1) == N && mget(v, other) == N) continue;
        if (mget(v, p - 1) != N) {
            out.push_back(Key{true, v});
        } else {
            out.push_back(Key{
                false,
                reverse ? blocked_exclude_reverse(v, W, p)
                        : blocked_exclude(v, p)});
        }
    }
    return out;
}

struct DiscoveryStats {
    std::uint64_t components = 0;
    std::uint64_t sources = 0;
    std::uint64_t destinations = 0;
    std::uint64_t edges = 0;
    FindStats destination_find;
    FindStats source_find;
    std::uint64_t max_sources = 0;
    std::uint64_t max_destinations = 0;
};

DiscoveryStats simulate_position(
    const std::vector<MateID>& labels, int W, int p, bool reverse, bool recent_first
) {
    DiscoveryStats st;
    const std::vector<Key> seeds = runtime_like_seeds(labels, W, p, reverse);
    for (const Key& seed : seeds) {
        ++st.components;
        std::vector<Key> src{seed};
        std::vector<Key> dst;
        std::size_t cursor = 0;
        while (cursor < src.size()) {
            const Key s = src[cursor++];
            const Vec edge = reduced_step_basis(s, W, p, reverse);
            for (const auto& [d, coef] : edge) {
                if (!coef) continue;
                ++st.edges;
                if (find_counted(dst, d, recent_first, st.destination_find) >= 0) continue;
                dst.push_back(d);
                const Vec pre = inverse_reduced(d, W, p, reverse);
                for (const auto& [x, a] : pre) {
                    if (!a) continue;
                    if (find_counted(src, x, recent_first, st.source_find) >= 0) continue;
                    src.push_back(x);
                }
            }
        }
        if (src.size() != dst.size()) fail("runtime find-work unbalanced component");
        st.sources += src.size();
        st.destinations += dst.size();
        st.max_sources = std::max<std::uint64_t>(st.max_sources, src.size());
        st.max_destinations = std::max<std::uint64_t>(st.max_destinations, dst.size());
    }
    return st;
}

void add_find(FindStats& a, const FindStats& b) {
    a.calls += b.calls;
    a.comparisons += b.comparisons;
    a.hits += b.hits;
    a.misses += b.misses;
    a.hit_comparisons += b.hit_comparisons;
    a.miss_comparisons += b.miss_comparisons;
    a.max_n = std::max(a.max_n, b.max_n);
}
void add(DiscoveryStats& a, const DiscoveryStats& b) {
    a.components += b.components;
    a.sources += b.sources;
    a.destinations += b.destinations;
    a.edges += b.edges;
    add_find(a.destination_find, b.destination_find);
    add_find(a.source_find, b.source_find);
    a.max_sources = std::max(a.max_sources, b.max_sources);
    a.max_destinations = std::max(a.max_destinations, b.max_destinations);
}

void print_mode(int W, const char* mode, const DiscoveryStats& st) {
    const auto total_calls = st.destination_find.calls + st.source_find.calls;
    const auto total_cmp = st.destination_find.comparisons + st.source_find.comparisons;
    const double cmp_per_component = st.components ? double(total_cmp) / double(st.components) : 0.0;
    const double cmp_per_find = total_calls ? double(total_cmp) / double(total_calls) : 0.0;
    std::cout << "W=" << W
              << " mode=" << mode
              << " components=" << st.components
              << " states=" << st.sources
              << " edges=" << st.edges
              << " find_calls=" << total_calls
              << " comparisons=" << total_cmp
              << " comparisons_per_component=" << std::fixed << std::setprecision(6)
              << cmp_per_component
              << " comparisons_per_find=" << cmp_per_find
              << " dst_calls=" << st.destination_find.calls
              << " dst_cmp=" << st.destination_find.comparisons
              << " dst_hits=" << st.destination_find.hits
              << " dst_misses=" << st.destination_find.misses
              << " src_calls=" << st.source_find.calls
              << " src_cmp=" << st.source_find.comparisons
              << " src_hits=" << st.source_find.hits
              << " src_misses=" << st.source_find.misses
              << " max_pairs=" << st.max_sources
              << " max_find_n=" << std::max(st.destination_find.max_n, st.source_find.max_n)
              << '\n';
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 11;
    if (maxW < 5 || maxW > 12) return 2;

    std::vector<std::vector<MateID>> words(std::size_t(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 5; W <= maxW; ++W) {
        DiscoveryStats forward_first;
        DiscoveryStats recent_first;
        for (bool reverse : {false, true}) {
            if (!reverse) {
                for (int p = W - 1; p >= 3; --p) {
                    add(forward_first, simulate_position(words[W - 1], W, p, false, false));
                    add(recent_first, simulate_position(words[W - 1], W, p, false, true));
                }
            } else {
                for (int p = 1; p <= W - 3; ++p) {
                    add(forward_first, simulate_position(words[W - 1], W, p, true, false));
                    add(recent_first, simulate_position(words[W - 1], W, p, true, true));
                }
            }
        }
        if (forward_first.components != recent_first.components ||
            forward_first.sources != recent_first.sources ||
            forward_first.edges != recent_first.edges)
            fail("runtime find-work traversal changed component totals");
        print_mode(W, "forward_first", forward_first);
        print_mode(W, "recent_first", recent_first);
        const auto fcmp = forward_first.destination_find.comparisons +
                          forward_first.source_find.comparisons;
        const auto rcmp = recent_first.destination_find.comparisons +
                          recent_first.source_find.comparisons;
        std::cout << "W=" << W
                  << " recent_vs_forward_comparison_ratio=" << std::fixed << std::setprecision(9)
                  << (fcmp ? double(rcmp) / double(fcmp) : 1.0)
                  << " better=" << (rcmp < fcmp ? 1 : 0) << '\n';
    }
    std::cout << "ALL_OK runtime_discovery_find_work_model=1\n";
    return 0;
}
