#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_tile_run_probe_main_unused
#include "gridfp_reduced_production_tile_run_probe.cpp"
#pragma pop_macro("main")

#include <numeric>

namespace {

std::uint32_t rotate_span_support(
    std::uint32_t support,
    int W,
    int K,
    bool reverse
) {
    const int span = 2 * K + 2;
    const int lo = reverse ? 0 : W - span;
    const std::uint32_t span_mask = ((std::uint32_t(1) << span) - 1u) << lo;
    const std::uint32_t x = (support & span_mask) >> lo;
    const int shift = reverse ? span - K : K;
    const std::uint32_t low_mask = (std::uint32_t(1) << span) - 1u;
    const std::uint32_t y = ((x << shift) | (x >> (span - shift))) & low_mask;
    return (support & ~span_mask) | (y << lo);
}

std::uint32_t swap_exclusive_support(
    std::uint32_t support,
    int W,
    int K,
    bool reverse
) {
    const int span = 2 * K + 2;
    const int lo = reverse ? 0 : W - span;
    const std::uint32_t k_mask = (std::uint32_t(1) << K) - 1u;
    const std::uint32_t low = (support >> lo) & k_mask;
    const std::uint32_t middle = (support >> (lo + K)) & 3u;
    const std::uint32_t high = (support >> (lo + K + 2)) & k_mask;
    const std::uint32_t span_mask = ((std::uint32_t(1) << span) - 1u) << lo;
    return (support & ~span_mask) |
           (high << lo) |
           (middle << (lo + K)) |
           (low << (lo + K + 2));
}

RunKey equal_tile_next_key(RunKey k, int W, int K, bool reverse) {
    k.support = k.blocked
        ? swap_exclusive_support(k.support, W, K, reverse)
        : rotate_span_support(k.support, W, K, reverse);
    return k;
}

struct EqualRunRec {
    int src_owner = -1;
    int dst_owner = -1;
    Rank src_base = 0;
    Rank dst_base = 0;
    Rank pc = 0;
};

struct EqualCycleStats {
    Rank runs = 0;
    Rank fixed_runs = 0;
    Rank nontrivial_cycles = 0;
    int max_cycle = 0;
};

EqualCycleStats verify_equal_tile_cycles(
    const std::vector<MateID>& main,
    const std::vector<MateID>& block,
    int W,
    int K,
    bool reverse,
    int ngpu
) {
    if (2 * K > W - 3) fail("equal tile exceeds interior");
    const int q = reverse ? 1 + K : W - 1 - K;
    const int old_start = reverse ? 1 : W - 1;
    const int new_start = q;
    ProductionFactorTables tables(W);
    const OwnerPlan plan = make_owner_plan(W, K, ngpu);

    std::map<RunKey, EqualRunRec> runs;
    for (Key key : layout(main, block, q)) {
        const MateID full = embed_full(key, W, q, reverse);
        const std::uint32_t support = occupancy_mask(full, W);
        const Rank pr = tables.primitive_rank(full, W);
        const Rank pc = catalan((__builtin_popcount(support) + 1) / 2);
        const GroupedRank sr = grouped_rank(
            key, tables, W, q, reverse, old_start, K, ngpu, plan);
        const GroupedRank dr = grouped_rank(
            key, tables, W, q, reverse, new_start, K, ngpu, plan);
        const RunKey rk{support, key.blocked};
        auto [it, inserted] = runs.emplace(rk, EqualRunRec{});
        EqualRunRec& z = it->second;
        if (inserted) {
            z.src_owner = sr.owner;
            z.dst_owner = dr.owner;
            z.src_base = sr.local - pr;
            z.dst_base = dr.local - pr;
            z.pc = pc;
        } else if (z.src_owner != sr.owner || z.dst_owner != dr.owner ||
                   z.src_base + pr != sr.local || z.dst_base + pr != dr.local || z.pc != pc) {
            fail("equal tile primitive run contiguity");
        }
    }

    std::map<std::tuple<int, Rank, Rank>, RunKey> source_interval;
    for (const auto& [rk, z] : runs) {
        const auto key = std::make_tuple(z.src_owner, z.src_base, z.pc);
        if (!source_interval.emplace(key, rk).second) fail("equal source run interval collision");
    }

    std::map<RunKey, RunKey> next;
    for (const auto& [rk, z] : runs) {
        const auto it = source_interval.find(std::make_tuple(z.dst_owner, z.dst_base, z.pc));
        if (it == source_interval.end()) fail("equal destination not an old run interval");
        const RunKey formula = equal_tile_next_key(rk, W, K, reverse);
        if (it->second.support != formula.support || it->second.blocked != formula.blocked)
            fail("equal tile run permutation formula");
        next.emplace(rk, formula);
    }

    EqualCycleStats st;
    st.runs = runs.size();
    std::set<RunKey> seen;
    const int main_order = (2 * K + 2) / std::gcd(2 * K + 2, K);
    for (const auto& [root, z] : runs) {
        (void)z;
        if (seen.count(root)) continue;
        RunKey cur = root;
        int len = 0;
        do {
            if (!seen.insert(cur).second) fail("equal cycle overlap");
            cur = next.at(cur);
            ++len;
            if (len > main_order + 2) fail("equal cycle did not close");
        } while (cur.support != root.support || cur.blocked != root.blocked);
        if (len == 1) ++st.fixed_runs;
        else ++st.nontrivial_cycles;
        st.max_cycle = std::max(st.max_cycle, len);
        if (root.blocked && len != 1 && len != 2) fail("blocked cycle order");
        if (!root.blocked && main_order % len != 0) fail("main cycle order");
    }
    if (seen.size() != runs.size()) fail("equal cycle coverage");
    return st;
}

RunTraffic equal_run_traffic_model(int W, int K, int ngpu) {
    if (ngpu < 2 || ngpu > 8) fail("equal traffic GPU range");
    const int common_bits = W - (2 * K + 2);
    if (common_bits < 0 || common_bits > 20) fail("equal traffic common bits");
    const int L = K + 2;
    RunTraffic out;
    const Rank common_count = Rank(1) << common_bits;
    for (Rank c = 0; c < common_count; ++c) {
        const RunOwnerHist old_hist = run_owner_hist_for_common(
            static_cast<std::uint32_t>(c), common_bits, K, L, ngpu);
        const RunOwnerHist new_hist = run_owner_hist_for_common(
            static_cast<std::uint32_t>(c), common_bits, K, L, ngpu);
        const int common_ones = __builtin_popcountll(c);
        for (int so = 0; so < ngpu; ++so) {
            for (int bo = 0; bo <= K; ++bo) {
                const Rank cb = old_hist.by_owner_ones[static_cast<std::size_t>(so)]
                                                      [static_cast<std::size_t>(bo)];
                if (!cb) continue;
                for (int d = 0; d < ngpu; ++d) {
                    for (int ao = 0; ao <= K; ++ao) {
                        const Rank ca = new_hist.by_owner_ones[static_cast<std::size_t>(d)]
                                                          [static_cast<std::size_t>(ao)];
                        if (!ca) continue;
                        const int base_occupied = common_ones + bo + ao;
                        const Rank state_weight = run_overlap_state_weight(base_occupied);
                        const Rank run_weight = (base_occupied & 1) ? 2 : 3;
                        const Rank pairs = cb * ca;
                        const Rank states = pairs * state_weight;
                        const Rank runs = pairs * run_weight;
                        out.total_states += states;
                        out.total_runs += runs;
                        out.pair_states[so][d] += states;
                        out.pair_runs[so][d] += runs;
                        if (so != d) {
                            out.moved_states += states;
                            out.moved_runs += runs;
                        }
                    }
                }
            }
        }
    }
    const Rank expected_runs = Rank(5) << (W - 3);
    if (out.total_runs != expected_runs) fail("equal traffic total runs");
    return out;
}

Rank w28_fixed_main_states_k12() {
    Rank total = 0;
    for (int a = 0; a <= 1; ++a) {
        for (int b = 0; b <= 1; ++b) {
            for (int t = 0; t <= 2; ++t) {
                const int occupied = 13 * (a + b) + t;
                if (!(occupied & 1)) continue;
                total += choose_u64(2, t) * catalan((occupied + 1) / 2);
            }
        }
    }
    return total;
}

Rank w28_fixed_blocked_states_k12() {
    Rank total = 0;
    for (int r = 0; r <= 12; ++r)
        total += choose_u64(12, r) * (catalan(r + 1) + catalan(r + 2));
    return total;
}

void print_w28_cycle_plan() {
    constexpr Rank main_states = 385719506620ULL;
    constexpr Rank blocked_states = 87677551081ULL;
    const Rank fixed_main = w28_fixed_main_states_k12();
    const Rank fixed_blocked = w28_fixed_blocked_states_k12();
    if ((main_states - fixed_main) % 13 != 0) fail("W28 main cycle divisibility");
    if ((blocked_states - fixed_blocked) % 2 != 0) fail("W28 blocked cycle divisibility");
    const Rank main_cycles = (main_states - fixed_main) / 13;
    const Rank blocked_cycles = (blocked_states - fixed_blocked) / 2;

    const RunTraffic traffic = equal_run_traffic_model(28, 12, 8);
    if (traffic.total_states != 473397057701ULL ||
        traffic.moved_states != 387344269008ULL ||
        traffic.moved_runs != 106060788ULL)
        fail("W28 equal traffic constants");
    std::cout << "W=28 scheme=12+12+fused1"
              << " redistribution_TiB=" << double(traffic.moved_states) * 4.0 / double(1ULL << 40)
              << " moved_fraction=" << double(traffic.moved_states) / double(traffic.total_states)
              << " moved_runs=" << traffic.moved_runs
              << " main_cycle_order=13 blocked_cycle_order=2"
              << " fixed_main_states=" << fixed_main
              << " fixed_blocked_states=" << fixed_blocked
              << " main_nontrivial_cycles=" << main_cycles
              << " blocked_nontrivial_cycles=" << blocked_cycles
              << " run_table_bytes=0 visited_bytes=0 scratch_bytes=0"
              << "\n";
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 11;
    const int ngpu = argc > 2 ? std::atoi(argv[2]) : 8;
    if (maxW < 7 || maxW > 12 || ngpu < 2 || ngpu > 8) return 2;

    std::vector<std::vector<MateID>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 7; W <= maxW; ++W) {
        for (int K = 2; 2 * K <= W - 3 && K <= 4; ++K) {
            for (bool reverse : {false, true}) {
                const EqualCycleStats s = verify_equal_tile_cycles(
                    words[W], words[W - 1], W, K, reverse, ngpu);
                const int order = (2 * K + 2) / std::gcd(2 * K + 2, K);
                std::cout << "W=" << W
                          << " K=" << K
                          << " direction=" << (reverse ? "reverse" : "forward")
                          << " runs=" << s.runs
                          << " fixed_runs=" << s.fixed_runs
                          << " nontrivial_cycles=" << s.nontrivial_cycles
                          << " max_cycle=" << s.max_cycle
                          << " main_order_bound=" << order
                          << " run_partition_preserved=1"
                          << " table_free_cycle_formula=1"
                          << " OK\n";
            }
        }
    }

    print_w28_cycle_plan();
    std::cout << "ALL_OK production_equal_tile_inplace_redistribution=1\n";
    return 0;
}
