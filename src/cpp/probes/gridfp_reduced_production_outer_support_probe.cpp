#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_factorized_probe_main_unused
#include "gridfp_reduced_production_factorized_probe.cpp"
#pragma pop_macro("main")

#include <deque>
#include <set>

namespace {

std::uint32_t occupancy_mask(MateID m, int W) {
    std::uint32_t z = 0;
    for (int bit = 0; bit < W; ++bit)
        if (mget(m, bit) != N) z |= std::uint32_t(1) << bit;
    return z;
}

MateID embed_full(Key k, int W, int p, bool reverse) {
    if (!k.blocked) return k.mate;
    return reverse ? blocked_exclude_reverse(k.mate, W, p)
                   : blocked_exclude(k.mate, p);
}

std::uint32_t outer_support(Key k, int W, int p, bool reverse) {
    const MateID full = embed_full(k, W, p, reverse);
    const std::uint32_t support = occupancy_mask(full, W);
    std::uint32_t local = 0;
    if (!reverse) {
        local |= std::uint32_t(1) << p;
        local |= std::uint32_t(1) << (p - 1);
        local |= std::uint32_t(1) << (p - 2);
    } else {
        local |= std::uint32_t(1) << (p - 1);
        local |= std::uint32_t(1) << p;
        local |= std::uint32_t(1) << (p + 1);
    }
    return support & ~local;
}

int occupied_sector(Key k, int W, int p, bool reverse) {
    const MateID full = embed_full(k, W, p, reverse);
    int occupied = 0;
    for (int bit = 0; bit < W; ++bit) occupied += mget(full, bit) != N;
    if (!(occupied & 1)) fail("outer-support occupied parity");
    return (occupied - 1) / 2;
}

std::vector<Key> component_seeds_outer(
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
            const MateID m = reverse
                ? blocked_exclude_reverse(v, W, p)
                : blocked_exclude(v, p);
            out.push_back(Key{false, m});
        }
    }
    return out;
}

struct OuterStats {
    Rank components = 0;
    Rank cross_shard = 0;
    Rank shard_sum = 0;
    Rank max_shards = 0;
    Rank one_sector = 0;
    Rank two_adjacent_sector = 0;
};

OuterStats verify_position_outer(
    const std::vector<MateID>& main,
    const std::vector<MateID>& block,
    const ProductionFactorTables& tables,
    int W,
    int p,
    bool reverse,
    int nshards
) {
    const int next = reverse ? p + 1 : p - 1;
    const ProductionFactorCodec src_codec(tables, p - 1);
    const ProductionFactorCodec dst_codec(tables, next - 1);
    const Rank n = src_codec.size();
    if (dst_codec.size() != n) fail("outer-support square layout");

    const auto seeds = component_seeds_outer(block, W, p, reverse);
    OuterStats st;
    st.components = seeds.size();

    std::set<Rank> source_covered;
    std::set<Rank> destination_covered;
    for (Key seed : seeds) {
        std::set<Key> sources;
        std::set<Key> destinations;
        std::deque<Key> queue;
        sources.insert(seed);
        queue.push_back(seed);
        while (!queue.empty()) {
            const Key s = queue.front();
            queue.pop_front();
            for (const auto& [d, c] : reduced_step_basis(s, W, p, reverse)) {
                if (c != 1 && c != -1) fail("outer-support coefficient");
                if (!destinations.insert(d).second) continue;
                for (const auto& [pre, a] : inverse_reduced(d, W, p, reverse)) {
                    if (a != 1 && a != -1) fail("outer-support inverse coefficient");
                    if (sources.insert(pre).second) queue.push_back(pre);
                }
            }
        }
        if (sources.size() != destinations.size()) fail("outer-support unbalanced component");

        bool have_outer = false;
        std::uint32_t want_outer = 0;
        std::set<int> sectors;
        std::set<int> shards;
        for (Key s : sources) {
            const std::uint32_t o = outer_support(s, W, p, reverse);
            if (!have_outer) { have_outer = true; want_outer = o; }
            else if (o != want_outer)
                fail(std::string(reverse ? "reverse" : "forward") +
                     " source outer support mismatch W=" + std::to_string(W) +
                     " p=" + std::to_string(p));
            sectors.insert(occupied_sector(s, W, p, reverse));
            const Rank r = src_codec.rank(s);
            source_covered.insert(r);
            const int shard = std::min<int>(nshards - 1, int((__uint128_t(r) * nshards) / n));
            shards.insert(shard);
        }
        for (Key d : destinations) {
            const std::uint32_t o = outer_support(d, W, next, reverse);
            if (!have_outer) { have_outer = true; want_outer = o; }
            else if (o != want_outer)
                fail(std::string(reverse ? "reverse" : "forward") +
                     " destination outer support mismatch W=" + std::to_string(W) +
                     " p=" + std::to_string(p));
            sectors.insert(occupied_sector(d, W, next, reverse));
            const Rank r = dst_codec.rank(d);
            destination_covered.insert(r);
            const int shard = std::min<int>(nshards - 1, int((__uint128_t(r) * nshards) / n));
            shards.insert(shard);
        }

        if (sectors.size() == 1) {
            ++st.one_sector;
        } else if (sectors.size() == 2) {
            auto it = sectors.begin();
            const int a = *it++;
            const int b = *it;
            if (b != a + 1) fail("nonadjacent sectors in component");
            ++st.two_adjacent_sector;
        } else {
            fail("component spans more than two sectors");
        }

        st.shard_sum += shards.size();
        st.max_shards = std::max<Rank>(st.max_shards, shards.size());
        if (shards.size() > 1) ++st.cross_shard;
    }

    if (source_covered.size() != n || destination_covered.size() != n)
        fail("outer-support component coverage");
    return st;
}

Rank catalan(int n) {
    std::vector<Rank> c(static_cast<std::size_t>(n + 1));
    c[0] = 1;
    for (int k = 1; k <= n; ++k) {
        __uint128_t z = 0;
        for (int i = 0; i < k; ++i) z += __uint128_t(c[i]) * c[k - 1 - i];
        c[k] = static_cast<Rank>(z);
    }
    return c[n];
}

Rank choose_u64(int n, int k) {
    if (k < 0 || k > n) return 0;
    if (k > n - k) k = n - k;
    __uint128_t z = 1;
    for (int i = 1; i <= k; ++i) z = z * (n - k + i) / i;
    return static_cast<Rank>(z);
}

// Number of reduced production coordinates sharing one fixed outer support of
// r occupied sites when the sliding local window has L physical sites.
Rank fixed_outer_group_size(int L, int r) {
    __uint128_t total = 0;
    for (int local = 0; local <= L; ++local) {
        const int occupied = r + local;
        if (!(occupied & 1)) continue;
        const int sector = (occupied - 1) / 2;
        const Rank pc = catalan(sector + 1);
        const Rank main_support = choose_u64(L, local);
        const Rank block_support = choose_u64(L - 2, local - 1);
        total += __uint128_t(main_support + block_support) * pc;
    }
    return static_cast<Rank>(total);
}

void print_w28_window_groups() {
    constexpr int W = 28;
    for (int K : {4, 6, 8, 10, 12}) {
        const int L = K + 2;
        const int outer = W - L;
        Rank largest = 0;
        int largest_r = -1;
        for (int r = 0; r <= outer; ++r) {
            const Rank z = fixed_outer_group_size(L, r);
            if (z > largest) { largest = z; largest_r = r; }
        }
        const double mib = double(largest) * 4.0 / double(1ULL << 20);
        std::cout << "W=28 window_steps=" << K
                  << " local_window=" << L
                  << " outer_bits=" << outer
                  << " largest_fixed_outer_group=" << largest
                  << " largest_outer_ones=" << largest_r
                  << " u32_group_MiB=" << mib
                  << "\n";
    }
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 9;
    const int nshards = argc > 2 ? std::atoi(argv[2]) : 8;
    if (maxW < 5 || maxW > 12 || nshards < 2 || nshards > 64) return 2;

    std::vector<std::vector<MateID>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 5; W <= maxW; ++W) {
        ProductionFactorTables tables(W);
        for (bool reverse : {false, true}) {
            const int begin = reverse ? 1 : W - 1;
            const int end = reverse ? W - 3 : 3;
            const int delta = reverse ? 1 : -1;
            for (int p = begin;; p += delta) {
                const OuterStats st = verify_position_outer(
                    words[W], words[W - 1], tables, W, p, reverse, nshards);
                const double cross = st.components ? double(st.cross_shard) / double(st.components) : 0.0;
                const double avg_shards = st.components ? double(st.shard_sum) / double(st.components) : 0.0;
                std::cout << "W=" << W
                          << " p=" << p
                          << " direction=" << (reverse ? "reverse" : "forward")
                          << " components=" << st.components
                          << " outer_support_invariant=1"
                          << " max_sector_span=2_adjacent"
                          << " one_sector=" << st.one_sector
                          << " two_adjacent_sector=" << st.two_adjacent_sector
                          << " contiguous_shards=" << nshards
                          << " cross_shard_fraction=" << cross
                          << " avg_shards_per_component=" << avg_shards
                          << " max_shards_per_component=" << st.max_shards
                          << "\n";
                if (p == end) break;
            }
        }
    }

    print_w28_window_groups();
    std::cout << "ALL_OK production_outer_support_locality=1"
              << " contiguous_rank_sharding_not_local=1"
              << " sliding_window_outer_support_candidate=1\n";
    return 0;
}
