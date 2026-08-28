#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_factorized_probe_main_unused
#include "gridfp_reduced_production_factorized_probe.cpp"
#pragma pop_macro("main")

#include <algorithm>
#include <deque>
#include <map>
#include <numeric>
#include <set>
#include <unordered_map>

namespace {

std::vector<Key> stationary_component_seeds(
    const std::vector<MateID>& labels,
    int W,
    int p,
    bool reverse
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
                        : blocked_exclude(v, p),
            });
        }
    }
    return out;
}

struct DSU {
    std::vector<Rank> parent;
    std::vector<Rank> size;

    explicit DSU(Rank n)
        : parent(static_cast<std::size_t>(n)),
          size(static_cast<std::size_t>(n), 1) {
        std::iota(parent.begin(), parent.end(), Rank(0));
    }

    Rank find(Rank x) {
        Rank r = x;
        while (parent[static_cast<std::size_t>(r)] != r)
            r = parent[static_cast<std::size_t>(r)];
        while (parent[static_cast<std::size_t>(x)] != x) {
            const Rank next = parent[static_cast<std::size_t>(x)];
            parent[static_cast<std::size_t>(x)] = r;
            x = next;
        }
        return r;
    }

    void unite(Rank a, Rank b) {
        a = find(a);
        b = find(b);
        if (a == b) return;
        if (size[static_cast<std::size_t>(a)] <
            size[static_cast<std::size_t>(b)])
            std::swap(a, b);
        parent[static_cast<std::size_t>(b)] = a;
        size[static_cast<std::size_t>(a)] += size[static_cast<std::size_t>(b)];
    }
};

struct PositionStats {
    Rank components = 0;
    Rank max_component = 0;
    Rank ranks_checked = 0;
};

PositionStats merge_position(
    DSU& dsu,
    const std::vector<MateID>& block_labels,
    const ProductionFactorTables& tables,
    int W,
    int p,
    bool reverse
) {
    const int next = reverse ? p + 1 : p - 1;
    ProductionFactorCodec src_codec(tables, p - 1);
    ProductionFactorCodec dst_codec(tables, next - 1);
    if (src_codec.size() != dst_codec.size() ||
        src_codec.size() != dsu.parent.size())
        fail("stationary supercomponent dimension");

    PositionStats stats;
    const auto seeds = stationary_component_seeds(block_labels, W, p, reverse);
    stats.components = seeds.size();

    for (Key seed : seeds) {
        std::set<Key> sources;
        std::set<Key> destinations;
        std::deque<Key> queue;
        sources.insert(seed);
        queue.push_back(seed);

        while (!queue.empty()) {
            const Key s = queue.front();
            queue.pop_front();
            for (const auto& [d, coef] : reduced_step_basis(s, W, p, reverse)) {
                if (coef != 1 && coef != -1)
                    fail("stationary supercomponent coefficient");
                if (!destinations.insert(d).second) continue;
                for (const auto& [pre, invcoef] : inverse_reduced(d, W, p, reverse)) {
                    if (invcoef != 1 && invcoef != -1)
                        fail("stationary supercomponent inverse coefficient");
                    if (sources.insert(pre).second) queue.push_back(pre);
                }
            }
        }
        if (sources.size() != destinations.size())
            fail("stationary supercomponent imbalance");

        std::vector<Rank> source_ranks;
        std::vector<Rank> dest_ranks;
        source_ranks.reserve(sources.size());
        dest_ranks.reserve(destinations.size());
        for (Key s : sources) source_ranks.push_back(src_codec.rank(s));
        for (Key d : destinations) dest_ranks.push_back(dst_codec.rank(d));
        std::sort(source_ranks.begin(), source_ranks.end());
        std::sort(dest_ranks.begin(), dest_ranks.end());
        if (source_ranks != dest_ranks)
            fail("stationary supercomponent factor-rank alignment");
        if (source_ranks.empty()) fail("stationary empty component");

        stats.max_component =
            std::max<Rank>(stats.max_component, source_ranks.size());
        stats.ranks_checked += source_ranks.size();
        const Rank root = source_ranks.front();
        for (std::size_t i = 1; i < source_ranks.size(); ++i)
            dsu.unite(root, source_ranks[i]);
    }
    if (stats.ranks_checked != src_codec.size())
        fail("stationary position coverage");
    return stats;
}

void report_supercomponents(DSU& dsu, int W, int positions) {
    std::unordered_map<Rank, Rank> counts;
    counts.reserve(dsu.parent.size());
    for (Rank r = 0; r < dsu.parent.size(); ++r)
        ++counts[dsu.find(r)];

    std::vector<Rank> sizes;
    sizes.reserve(counts.size());
    for (const auto& [_, n] : counts) sizes.push_back(n);
    std::sort(sizes.begin(), sizes.end(), std::greater<Rank>());

    const Rank states = dsu.parent.size();
    const Rank largest = sizes.empty() ? 0 : sizes.front();
    Rank top8 = 0;
    for (std::size_t i = 0; i < std::min<std::size_t>(8, sizes.size()); ++i)
        top8 += sizes[i];

    std::cout << "production-stationary-supercomponents"
              << " W=" << W
              << " states=" << states
              << " positions=" << positions
              << " supercomponents=" << sizes.size()
              << " largest=" << largest
              << " largest_fraction="
              << (states ? double(largest) / double(states) : 0.0)
              << " top8_fraction="
              << (states ? double(top8) / double(states) : 0.0)
              << " fixed_component_invariant_owner_max_shards=" << sizes.size()
              << '\n';
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 11;
    if (maxW < 5 || maxW > 12) return 2;

    std::vector<std::vector<MateID>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 5; W <= maxW; ++W) {
        ProductionFactorTables tables(W);
        ProductionFactorCodec ref_codec(tables, 1);
        DSU dsu(ref_codec.size());
        int positions = 0;
        Rank max_local_component = 0;

        for (int p = W - 1; p >= 3; --p) {
            const auto s = merge_position(
                dsu, words[W - 1], tables, W, p, false);
            max_local_component = std::max(max_local_component, s.max_component);
            ++positions;
        }
        for (int p = 1; p <= W - 3; ++p) {
            const auto s = merge_position(
                dsu, words[W - 1], tables, W, p, true);
            max_local_component = std::max(max_local_component, s.max_component);
            ++positions;
        }

        report_supercomponents(dsu, W, positions);
        std::cout << "production-stationary-local-component"
                  << " W=" << W
                  << " max_one_step_component=" << max_local_component
                  << '\n';
    }

    std::cout << "ALL_OK production_stationary_supercomponent_probe=1\n";
    return 0;
}
