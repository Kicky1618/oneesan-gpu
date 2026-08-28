#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_outer_support_probe_main_unused
#include "gridfp_reduced_production_outer_support_probe.cpp"
#pragma pop_macro("main")

#include <deque>
#include <iomanip>
#include <limits>
#include <set>

namespace {

std::uint32_t compact_outside_tile(MateID full, int W, int lo, int hi) {
    std::uint32_t out = 0;
    int q = 0;
    for (int bit = 0; bit < W; ++bit) {
        if (bit >= lo && bit <= hi) continue;
        if (mget(full, bit) != N) out |= std::uint32_t(1) << q;
        ++q;
    }
    return out;
}

std::uint32_t tile_outer_support_host(
    Key k,
    int W,
    int q,
    bool reverse,
    int lo,
    int hi
) {
    return compact_outside_tile(embed_full(k, W, q, reverse), W, lo, hi);
}

Rank compact_support_rank_host(std::uint32_t mask, int len, int ones) {
    Rank rank = 0;
    int left = ones;
    for (int pos = 0; pos < len; ++pos) {
        if (((mask >> pos) & 1u) == 0) continue;
        const int rem = len - pos - 1;
        rank += choose_u64(rem, left);
        --left;
    }
    if (left != 0) fail("tile owner support rank");
    return rank;
}

Rank tile_group_total(int L, int O) {
    Rank total = 0;
    for (int r = 0; r <= O; ++r)
        total += choose_u64(O, r) * fixed_outer_group_size(L, r);
    return total;
}

int weighted_outer_owner_host(
    std::uint32_t compact_outer,
    int L,
    int O,
    int ngpu
) {
    const int r = __builtin_popcount(compact_outer);
    const Rank group = fixed_outer_group_size(L, r);
    const Rank sr = compact_support_rank_host(compact_outer, O, r);
    Rank prefix = 0;
    for (int t = 0; t < r; ++t)
        prefix += choose_u64(O, t) * fixed_outer_group_size(L, t);
    const Rank total = tile_group_total(L, O);
    const Rank midpoint = prefix + sr * group + group / 2;
    int owner = int((__uint128_t(midpoint) * ngpu) / total);
    if (owner >= ngpu) owner = ngpu - 1;
    return owner;
}

std::vector<Rank> theoretical_owner_loads(int W, int K, int ngpu) {
    const int L = K + 2;
    const int O = W - L;
    if (O < 0) fail("tile owner negative outer width");
    const Rank total = tile_group_total(L, O);
    std::vector<Rank> load(static_cast<std::size_t>(ngpu));
    Rank prefix = 0;
    for (int r = 0; r <= O; ++r) {
        const Rank group = fixed_outer_group_size(L, r);
        const Rank count = choose_u64(O, r);
        for (Rank sr = 0; sr < count; ++sr) {
            const Rank midpoint = prefix + sr * group + group / 2;
            int owner = int((__uint128_t(midpoint) * ngpu) / total);
            if (owner >= ngpu) owner = ngpu - 1;
            load[static_cast<std::size_t>(owner)] += group;
        }
        prefix += count * group;
    }
    return load;
}

struct TileStepStats {
    Rank components = 0;
    Rank states = 0;
    Rank owner_crossings = 0;
    std::vector<Rank> owner_load;
};

TileStepStats verify_tile_step(
    const std::vector<MateID>& main,
    const std::vector<MateID>& block,
    int W,
    int p,
    bool reverse,
    int tile_start,
    int K,
    int ngpu
) {
    const int next = reverse ? p + 1 : p - 1;
    const int L = K + 2;
    const int lo = reverse ? tile_start - 1 : tile_start - K - 1;
    const int hi = reverse ? tile_start + K : tile_start;
    const int O = W - L;
    if (lo < 0 || hi >= W || O < 0 || hi - lo + 1 != L)
        fail("tile owner invalid geometry");

    const auto seeds = component_seeds_outer(block, W, p, reverse);
    TileStepStats st;
    st.components = seeds.size();
    st.owner_load.assign(static_cast<std::size_t>(ngpu), 0);

    std::set<Key> source_covered;
    std::set<Key> destination_covered;
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
                if (c != 1 && c != -1) fail("tile owner coefficient");
                if (!destinations.insert(d).second) continue;
                for (const auto& [pre, a] : inverse_reduced(d, W, p, reverse)) {
                    if (a != 1 && a != -1) fail("tile owner inverse coefficient");
                    if (sources.insert(pre).second) queue.push_back(pre);
                }
            }
        }
        if (sources.size() != destinations.size()) fail("tile owner component imbalance");

        bool have = false;
        std::uint32_t outer = 0;
        int owner = -1;
        for (Key s : sources) {
            const std::uint32_t z = tile_outer_support_host(s, W, p, reverse, lo, hi);
            if (!have) {
                have = true;
                outer = z;
                owner = weighted_outer_owner_host(outer, L, O, ngpu);
            } else if (z != outer) {
                fail("source escaped fixed tile outer support");
            }
            source_covered.insert(s);
        }
        for (Key d : destinations) {
            const std::uint32_t z = tile_outer_support_host(d, W, next, reverse, lo, hi);
            if (!have || z != outer) {
                ++st.owner_crossings;
                fail("destination escaped fixed tile outer support");
            }
            const int d_owner = weighted_outer_owner_host(z, L, O, ngpu);
            if (d_owner != owner) {
                ++st.owner_crossings;
                fail("component crossed fixed tile GPU owner");
            }
            destination_covered.insert(d);
        }
        st.owner_load[static_cast<std::size_t>(owner)] += sources.size();
        st.states += sources.size();
    }

    const Rank expected_states = main.size() + block.size() - gen_words(W - 2).size();
    if (st.states != expected_states || source_covered.size() != expected_states ||
        destination_covered.size() != expected_states)
        fail("tile owner state coverage");
    const auto theory = theoretical_owner_loads(W, K, ngpu);
    if (st.owner_load != theory)
        fail("tile owner load differs from closed-form grouping");
    return st;
}

void verify_tile(
    const std::vector<MateID>& main,
    const std::vector<MateID>& block,
    int W,
    bool reverse,
    int tile_start,
    int K,
    int ngpu
) {
    const int first = tile_start;
    const int last = reverse ? tile_start + K - 1 : tile_start - K + 1;
    std::vector<Rank> reference_load;
    Rank states = 0;
    Rank components = 0;
    for (int p = first;; p += reverse ? 1 : -1) {
        const TileStepStats st = verify_tile_step(
            main, block, W, p, reverse, tile_start, K, ngpu);
        if (reference_load.empty()) {
            reference_load = st.owner_load;
            states = st.states;
            components = st.components;
        } else if (st.owner_load != reference_load || st.states != states || st.components != components) {
            fail("tile owner load changed inside tile");
        }
        if (p == last) break;
    }
}

void print_w28_plan(int K, int ngpu) {
    constexpr int W = 28;
    const int L = K + 2;
    const int O = W - L;
    const auto load = theoretical_owner_loads(W, K, ngpu);
    const Rank total = tile_group_total(L, O);
    Rank max_load = 0;
    Rank min_load = std::numeric_limits<Rank>::max();
    Rank max_group = 0;
    for (Rank x : load) {
        max_load = std::max(max_load, x);
        min_load = std::min(min_load, x);
    }
    for (int r = 0; r <= O; ++r)
        max_group = std::max(max_group, fixed_outer_group_size(L, r));

    const double gib = 4.0 / double(1ULL << 30);
    const double avg = double(total) / ngpu;
    const double b300_gib = 288000000000.0 / double(1ULL << 30);
    std::cout << std::fixed << std::setprecision(6)
              << "W=28_tile_plan"
              << " K=" << K
              << " local_window=" << L
              << " outer_bits=" << O
              << " groups=" << (Rank(1) << O)
              << " total_states=" << total
              << " max_group_GiB=" << double(max_group) * gib
              << " min_gpu_GiB=" << double(min_load) * gib
              << " max_gpu_GiB=" << double(max_load) * gib
              << " imbalance=" << double(max_load) / avg
              << " B300_288GB_GiB=" << b300_gib
              << " worst_gpu_headroom_GiB=" << b300_gib - double(max_load) * gib
              << '\n';
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 9;
    const int ngpu = argc > 2 ? std::atoi(argv[2]) : 8;
    if (maxW < 5 || maxW > 11 || ngpu < 2 || ngpu > 64) return 2;

    std::vector<std::vector<MateID>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 5; W <= maxW; ++W) {
        const int interior = W - 3;
        const int maxK = std::min(4, interior);
        for (int K = 1; K <= maxK; ++K) {
            for (int start = W - 1; start - K + 1 >= 3; --start)
                verify_tile(words[W], words[W - 1], W, false, start, K, ngpu);
            for (int start = 1; start + K - 1 <= W - 3; ++start)
                verify_tile(words[W], words[W - 1], W, true, start, K, ngpu);
        }
        std::cout << "W=" << W
                  << " tile_K_tested=1.." << maxK
                  << " ngpu=" << ngpu
                  << " fixed_outer_support=1"
                  << " owner_crossings=0"
                  << " closed_form_owner_load=OK"
                  << " forward=OK reverse=OK\n";
    }

    print_w28_plan(9, ngpu);
    print_w28_plan(12, ngpu);
    print_w28_plan(16, ngpu);
    print_w28_plan(18, ngpu);
    std::cout << "W=28_schedule interior_steps=25"
              << " tile0_K=16 tile1_K=9"
              << " interior_redistributions_per_row=1"
              << " row_edges_separate=1\n";
    std::cout << "ALL_OK production_multistep_outer_owner=1\n";
    return 0;
}
