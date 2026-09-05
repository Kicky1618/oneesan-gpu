#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_outer_support_probe_main_unused
#include "gridfp_reduced_production_outer_support_probe.cpp"
#pragma pop_macro("main")

namespace {

Rank support_rank_fixed(std::uint32_t mask, int len, int ones) {
    Rank rank = 0;
    int left = ones;
    for (int pos = 0; pos < len; ++pos) {
        if (((mask >> pos) & 1u) == 0) continue;
        const int rem = len - pos - 1;
        rank += choose_u64(rem, left);
        --left;
    }
    if (left != 0) fail("owner support rank");
    return rank;
}

std::uint32_t compress_outside_window(
    std::uint32_t full,
    int W,
    int lo,
    int hi
) {
    std::uint32_t compact = 0;
    int q = 0;
    for (int bit = 0; bit < W; ++bit) {
        if (bit >= lo && bit <= hi) continue;
        if ((full >> bit) & 1u) compact |= std::uint32_t(1) << q;
        ++q;
    }
    return compact;
}

Rank group_prefix_before_r(int L, int O, int r) {
    __uint128_t z = 0;
    for (int t = 0; t < r; ++t)
        z += __uint128_t(choose_u64(O, t)) * fixed_outer_group_size(L, t);
    return static_cast<Rank>(z);
}

Rank total_grouped_states(int L, int O) {
    __uint128_t z = 0;
    for (int r = 0; r <= O; ++r)
        z += __uint128_t(choose_u64(O, r)) * fixed_outer_group_size(L, r);
    return static_cast<Rank>(z);
}

int weighted_owner(
    std::uint32_t compact_outer,
    int L,
    int O,
    int ngpu
) {
    const int r = __builtin_popcount(compact_outer);
    const Rank group = fixed_outer_group_size(L, r);
    const Rank sr = support_rank_fixed(compact_outer, O, r);
    const Rank prefix = group_prefix_before_r(L, O, r);
    const Rank total = total_grouped_states(L, O);
    const __uint128_t midpoint = __uint128_t(prefix) + __uint128_t(sr) * group + group / 2;
    int owner = int(midpoint * ngpu / total);
    if (owner >= ngpu) owner = ngpu - 1;
    return owner;
}

std::uint32_t full_support_for_q(Key k, int W, int q, bool reverse) {
    return occupancy_mask(embed_full(k, W, q, reverse), W);
}

void verify_tile_edges(
    const std::vector<MateID>& main,
    const std::vector<MateID>& block,
    int W,
    int start,
    int K,
    bool reverse,
    int ngpu
) {
    const int L = K + 2;
    const int O = W - L;
    const int lo = reverse ? start - 1 : start - K - 1;
    const int hi = reverse ? start + K : start;
    if (lo < 0 || hi >= W || hi - lo + 1 != L) fail("owner tile window");

    int q = start;
    for (int step = 0; step < K; ++step) {
        const int next = reverse ? q + 1 : q - 1;
        for (Key src : layout(main, block, q)) {
            const std::uint32_t ss = full_support_for_q(src, W, q, reverse);
            const std::uint32_t so = compress_outside_window(ss, W, lo, hi);
            const int owner = weighted_owner(so, L, O, ngpu);
            for (const auto& [dst, c] : reduced_step_basis(src, W, q, reverse)) {
                if (c != 1 && c != -1) fail("owner tile coefficient");
                const std::uint32_t ds = full_support_for_q(dst, W, next, reverse);
                const std::uint32_t dout = compress_outside_window(ds, W, lo, hi);
                if (dout != so)
                    fail(std::string(reverse ? "reverse" : "forward") +
                         " tile outer support changed W=" + std::to_string(W) +
                         " start=" + std::to_string(start) +
                         " K=" + std::to_string(K) +
                         " q=" + std::to_string(q));
                if (weighted_owner(dout, L, O, ngpu) != owner)
                    fail("owner changed inside tile");
            }
        }
        q = next;
    }
}

struct LoadReport {
    std::vector<Rank> states;
    std::vector<Rank> groups;
};

LoadReport exact_w28_owner_load(int K, int ngpu) {
    constexpr int W = 28;
    const int L = K + 2;
    const int O = W - L;
    const Rank total = total_grouped_states(L, O);
    LoadReport out;
    out.states.assign(static_cast<std::size_t>(ngpu), 0);
    out.groups.assign(static_cast<std::size_t>(ngpu), 0);
    Rank prefix = 0;
    for (int r = 0; r <= O; ++r) {
        const Rank g = fixed_outer_group_size(L, r);
        const Rank count = choose_u64(O, r);
        for (Rank sr = 0; sr < count; ++sr) {
            const __uint128_t midpoint = __uint128_t(prefix) + __uint128_t(sr) * g + g / 2;
            int owner = int(midpoint * ngpu / total);
            if (owner >= ngpu) owner = ngpu - 1;
            out.states[static_cast<std::size_t>(owner)] += g;
            ++out.groups[static_cast<std::size_t>(owner)];
        }
        prefix += count * g;
    }
    Rank sum = 0;
    for (Rank z : out.states) sum += z;
    if (sum != total || total != 473397057701ULL) fail("W28 owner load total");
    return out;
}

void print_w28_owner_loads(int ngpu) {
    for (int K : {4, 6, 8, 10, 12}) {
        const LoadReport r = exact_w28_owner_load(K, ngpu);
        const auto [lo, hi] = std::minmax_element(r.states.begin(), r.states.end());
        const double spread_gib = double(*hi - *lo) * 4.0 / double(1ULL << 30);
        std::cout << "W=28 owner_window_steps=" << K
                  << " ngpu=" << ngpu
                  << " min_states=" << *lo
                  << " max_states=" << *hi
                  << " min_GiB=" << double(*lo) * 4.0 / double(1ULL << 30)
                  << " max_GiB=" << double(*hi) * 4.0 / double(1ULL << 30)
                  << " spread_GiB=" << spread_gib
                  << " whole_group_owner=1 table_bytes=0\n";
    }
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 10;
    const int ngpu = argc > 2 ? std::atoi(argv[2]) : 8;
    if (maxW < 7 || maxW > 12 || ngpu < 2 || ngpu > 64) return 2;

    std::vector<std::vector<MateID>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 7; W <= maxW; ++W) {
        for (int K = 2; K <= std::min(4, W - 3); ++K) {
            for (int start = K + 2; start <= W - 1; ++start)
                verify_tile_edges(words[W], words[W - 1], W, start, K, false, ngpu);
            for (int start = 1; start <= W - K - 2; ++start)
                verify_tile_edges(words[W], words[W - 1], W, start, K, true, ngpu);
            std::cout << "W=" << W
                      << " K=" << K
                      << " local_window=" << (K + 2)
                      << " tile_outer_support_invariant=1"
                      << " owner_constant_inside_tile=1"
                      << " ngpu=" << ngpu
                      << " forward=OK reverse=OK\n";
        }
    }

    print_w28_owner_loads(ngpu);
    std::cout << "ALL_OK production_multigpu_outer_owner=1"
              << " weighted_prefix_owner=1"
              << " no_intra_tile_redistribution=1\n";
    return 0;
}
