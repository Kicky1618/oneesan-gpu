#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_grouped_codec_probe_main_unused
#include "gridfp_reduced_production_grouped_codec_probe.cpp"
#pragma pop_macro("main")

#include <array>
#include <set>

namespace {

Vec projected_step_vec(const Vec& in, int W, int p, bool reverse) {
    const int next = reverse ? p + 1 : p - 1;
    return project_vec(step_vec(in, W, p, reverse), W, next, reverse);
}

Vec raw_two_rows(MateID m, int W) {
    Vec v;
    add(v, Key{false, m}, 1);
    for (int p = W - 1; p >= 1; --p) v = step_vec(v, W, p, false);
    for (int p = 1; p < W; ++p) v = step_vec(v, W, p, true);
    return v;
}

Vec scheduled_two_rows(MateID m, int W, int K) {
    Vec v;
    add(v, Key{false, m}, 1);

    // Enter the high Q channel with the first forward edge already consumed.
    v = project_vec(step_vec(v, W, W - 1, false), W, W - 2, false);
    // High window [K, W-1], K-1 reduced updates: Q_{W-2} -> Q_{K+1}.
    for (int p = W - 2; p >= K + 2; --p) v = projected_step_vec(v, W, p, false);
    // Same Q_{K+1} coordinates are redistributed in-place to low window [0,K+1].
    // Low window performs K updates Q_{K+1} -> Q_1.
    for (int p = K + 1; p >= 2; --p) v = projected_step_vec(v, W, p, false);
    // Low turn compression to main.
    v = step_vec(v, W, 1, false);

    // Low turn expansion starts the reverse row and lands directly in Q_2.
    v = project_vec(step_vec(v, W, 1, true), W, 2, true);
    // Low reverse window: K-1 updates Q_2 -> Q_{K+1}.
    for (int p = 2; p <= K; ++p) v = projected_step_vec(v, W, p, true);
    // Redistribute the same Q_{K+1} coordinates back to high window.
    // High reverse window: K updates Q_{K+1} -> Q_{W-1}.
    for (int p = K + 1; p <= W - 2; ++p) v = projected_step_vec(v, W, p, true);
    // High turn compression back to main.
    v = step_vec(v, W, W - 1, true);
    return v;
}

void verify_tile_owner_edges(
    const std::vector<MateID>& main,
    const std::vector<MateID>& block,
    const ProductionFactorTables& tables,
    int W,
    int K,
    bool reverse,
    bool high,
    int ngpu
) {
    const OwnerPlan plan = make_owner_plan(W, K, ngpu);
    const int tile_start = !reverse
        ? (high ? W - 1 : K + 1)
        : (high ? K + 1 : 1);
    const int begin = !reverse
        ? (high ? W - 2 : K + 1)
        : (high ? K + 1 : 2);
    const int end = !reverse
        ? (high ? K + 2 : 2)
        : (high ? W - 2 : K);
    const int delta = reverse ? 1 : -1;

    for (int p = begin;; p += delta) {
        const int next = reverse ? p + 1 : p - 1;
        const auto src = layout(main, block, p);
        for (Key k : src) {
            const GroupedRank sr = grouped_rank(
                k, tables, W, p, reverse, tile_start, K, ngpu, plan);
            for (const auto& [d, c] : reduced_step_basis(k, W, p, reverse)) {
                if (c != 1 && c != -1) fail("schedule tile coefficient");
                const GroupedRank dr = grouped_rank(
                    d, tables, W, next, reverse, tile_start, K, ngpu, plan);
                if (sr.owner != dr.owner) fail("schedule intra-tile owner crossing");
            }
        }
        if (p == end) break;
    }
}

void verify_low_turn_slots(
    const std::vector<MateID>& main_words,
    const std::vector<MateID>& block_words,
    const ProductionFactorTables& tables,
    int W,
    int K,
    int ngpu
) {
    const OwnerPlan plan = make_owner_plan(W, K, ngpu);
    const std::vector<Key> q1 = layout(main_words, block_words, 1);
    const std::vector<Key> q2 = layout(main_words, block_words, 2);
    std::vector<Key> main;
    for (MateID m : main_words) main.push_back(Key{false, m});

    const int fstart = K + 1; // [0,K+1]
    const int rstart = 1;     // exactly the same physical window [0,K+1]

    // Main coordinates are direction-independent when the physical window is equal.
    for (Key k : main) {
        const GroupedRank a = grouped_rank(k, tables, W, 1, false, fstart, K, ngpu, plan);
        const GroupedRank b = grouped_rank(k, tables, W, 1, true, rstart, K, ngpu, plan);
        if (a.owner != b.owner || a.local != b.local) fail("turn main slot direction mismatch");
    }

    // Forward p1 compression: every destination main slot is one of the source
    // component slots, so all sources can be loaded before overwriting destinations.
    std::map<Key, Rank> main_rank;
    for (Rank i = 0; i < main.size(); ++i) main_rank.emplace(main[static_cast<std::size_t>(i)], i);
    const Rank ns = q1.size(), nm = main.size();
    std::vector<std::vector<Rank>> adj(static_cast<std::size_t>(ns + nm));
    for (Rank s = 0; s < ns; ++s) {
        for (const auto& [d, c] : step_basis(q1[static_cast<std::size_t>(s)], W, 1, false)) {
            if (!c) fail("turn compression zero");
            const Rank j = main_rank.at(d);
            adj[static_cast<std::size_t>(s)].push_back(ns + j);
            adj[static_cast<std::size_t>(ns + j)].push_back(s);
        }
    }
    std::vector<std::uint8_t> seen(static_cast<std::size_t>(ns + nm));
    Rank max_compress = 0;
    for (Rank root = 0; root < ns + nm; ++root) {
        if (seen[static_cast<std::size_t>(root)]) continue;
        std::vector<Rank> stack{root}, ss, dd;
        seen[static_cast<std::size_t>(root)] = 1;
        while (!stack.empty()) {
            const Rank x = stack.back(); stack.pop_back();
            if (x < ns) ss.push_back(x); else dd.push_back(x - ns);
            for (Rank y : adj[static_cast<std::size_t>(x)]) if (!seen[static_cast<std::size_t>(y)]) {
                seen[static_cast<std::size_t>(y)] = 1; stack.push_back(y);
            }
        }
        std::set<std::pair<int,Rank>> S,D;
        for (Rank s : ss) {
            const auto r = grouped_rank(q1[static_cast<std::size_t>(s)], tables, W, 1, false, fstart, K, ngpu, plan);
            S.emplace(r.owner,r.local);
        }
        for (Rank d : dd) {
            const auto r = grouped_rank(main[static_cast<std::size_t>(d)], tables, W, 1, false, fstart, K, ngpu, plan);
            D.emplace(r.owner,r.local);
        }
        if (!std::includes(S.begin(),S.end(),D.begin(),D.end())) fail("turn compression slot subset");
        max_compress = std::max<Rank>(max_compress, ss.size());
    }

    // Reverse p1 expansion: source main slots are a subset of destination Q2
    // component slots in the same physical window.
    std::map<Key, Rank> q2_rank;
    for (Rank i = 0; i < q2.size(); ++i) q2_rank.emplace(q2[static_cast<std::size_t>(i)], i);
    std::vector<std::vector<Rank>> radj(static_cast<std::size_t>(nm + q2.size()));
    for (Rank s = 0; s < nm; ++s) {
        Vec col = project_vec(step_basis(main[static_cast<std::size_t>(s)], W, 1, true), W, 2, true);
        for (const auto& [d,c] : col) {
            if (!c) fail("turn expansion zero");
            const Rank j = q2_rank.at(d);
            radj[static_cast<std::size_t>(s)].push_back(nm + j);
            radj[static_cast<std::size_t>(nm + j)].push_back(s);
        }
    }
    seen.assign(static_cast<std::size_t>(nm + q2.size()), 0);
    Rank max_expand = 0;
    for (Rank root = 0; root < nm + q2.size(); ++root) {
        if (seen[static_cast<std::size_t>(root)]) continue;
        std::vector<Rank> stack{root}, ss, dd;
        seen[static_cast<std::size_t>(root)] = 1;
        while (!stack.empty()) {
            const Rank x = stack.back(); stack.pop_back();
            if (x < nm) ss.push_back(x); else dd.push_back(x - nm);
            for (Rank y : radj[static_cast<std::size_t>(x)]) if (!seen[static_cast<std::size_t>(y)]) {
                seen[static_cast<std::size_t>(y)] = 1; stack.push_back(y);
            }
        }
        std::set<std::pair<int,Rank>> S,D;
        for (Rank s : ss) {
            const auto r = grouped_rank(main[static_cast<std::size_t>(s)], tables, W, 1, true, rstart, K, ngpu, plan);
            S.emplace(r.owner,r.local);
        }
        for (Rank d : dd) {
            const auto r = grouped_rank(q2[static_cast<std::size_t>(d)], tables, W, 2, true, rstart, K, ngpu, plan);
            D.emplace(r.owner,r.local);
        }
        if (!std::includes(D.begin(),D.end(),S.begin(),S.end())) fail("turn expansion slot subset");
        max_expand = std::max<Rank>(max_expand, dd.size());
    }

    std::cout << "W=" << W << " K=" << K
              << " low_turn_same_window=1"
              << " main_slot_identity=1"
              << " compress_max_src=" << max_compress
              << " expand_max_dst=" << max_expand
              << " second_state_buffer_bytes=0\n";
}

Rank overlap_weight(int base_occupied) {
    if (base_occupied & 1)
        return catalan((base_occupied + 1) / 2) + catalan((base_occupied + 3) / 2);
    return 3 * catalan((base_occupied + 2) / 2);
}

std::array<std::array<Rank,14>,8> owner_hist_k13(int ngpu) {
    if (ngpu != 8) fail("W28 schedule histogram fixed to 8 GPUs");
    std::array<std::array<Rank,14>,8> h{};
    constexpr int K = 13;
    constexpr int L = 15;
    constexpr int O = 13;
    for (std::uint32_t mask = 0; mask < (1u << K); ++mask) {
        const int r = __builtin_popcount(mask);
        const int owner = weighted_owner(mask, L, O, ngpu);
        ++h[static_cast<std::size_t>(owner)][static_cast<std::size_t>(r)];
    }
    return h;
}

void print_w28_plan() {
    constexpr int W = 28, K = 13, ngpu = 8;
    const OwnerPlan p = make_owner_plan(W, K, ngpu);
    Rank lo = std::numeric_limits<Rank>::max(), hi = 0;
    for (Rank z : p.size) { lo = std::min(lo,z); hi = std::max(hi,z); }

    const auto h = owner_hist_k13(ngpu);
    Rank total = 0, moved = 0;
    for (int so = 0; so < ngpu; ++so) for (int a = 0; a <= K; ++a) {
        const Rank ca = h[static_cast<std::size_t>(so)][static_cast<std::size_t>(a)];
        if (!ca) continue;
        for (int d = 0; d < ngpu; ++d) for (int b = 0; b <= K; ++b) {
            const Rank cb = h[static_cast<std::size_t>(d)][static_cast<std::size_t>(b)];
            if (!cb) continue;
            const Rank term = ca * cb * overlap_weight(a + b);
            total += term;
            if (so != d) moved += term;
        }
    }
    if (total != 473397057701ULL) fail("W28 two-window total");

    std::cout << "W=28 K=13"
              << " low_window=[0,14] high_window=[13,27] overlap_bits=2"
              << " owner_min_GiB=" << double(lo) * 4.0 / double(1ULL<<30)
              << " owner_max_GiB=" << double(hi) * 4.0 / double(1ULL<<30)
              << " redistribution_states=" << moved
              << " redistribution_fraction=" << double(moved)/double(total)
              << " redistribution_TiB=" << double(moved)*4.0/double(1ULL<<40)
              << " main_cycle_order=28 blocked_cycle_order=2"
              << " redistributions_per_row=1"
              << " run_table_bytes=0 visited_bytes=0"
              << " state_streams_per_gpu=1\n";
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 12;
    const int ngpu = argc > 2 ? std::atoi(argv[2]) : 8;
    if (maxW < 6 || maxW > 12 || ngpu != 8) return 2;

    for (int W = 6; W <= maxW; W += 2) {
        const int K = (W - 2) / 2;
        const auto main = gen_words(W);
        const auto block = gen_words(W - 1);
        ProductionFactorTables tables(W);

        for (MateID m : main) {
            if (scheduled_two_rows(m, W, K) != raw_two_rows(m, W))
                fail("two-window two-row operator mismatch W=" + std::to_string(W));
        }
        verify_tile_owner_edges(main, block, tables, W, K, false, true, ngpu);
        verify_tile_owner_edges(main, block, tables, W, K, false, false, ngpu);
        verify_tile_owner_edges(main, block, tables, W, K, true, false, ngpu);
        verify_tile_owner_edges(main, block, tables, W, K, true, true, ngpu);
        verify_low_turn_slots(main, block, tables, W, K, ngpu);
        std::cout << "W=" << W << " K=" << K
                  << " two_window_schedule_exact=1"
                  << " intra_tile_owner_crossings=0"
                  << " one_redistribution_per_row=1"
                  << " forward_reverse=OK\n";
    }

    print_w28_plan();
    std::cout << "ALL_OK production_two_window_single_stream_schedule=1\n";
    return 0;
}
