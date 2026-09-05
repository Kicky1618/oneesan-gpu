#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_shift_cycle_probe_main_unused
#include "gridfp_reduced_production_shift_cycle_probe.cpp"
#pragma pop_macro("main")

#include <numeric>

namespace {

struct MainRunRec {
    int src_owner = -1;
    int dst_owner = -1;
    Rank src_base = 0;
    Rank dst_base = 0;
    Rank pc = 0;
};

struct MainShiftStats {
    Rank states = 0;
    Rank runs = 0;
    Rank cycles = 0;
    Rank fixed_runs = 0;
    int max_cycle = 0;
};

MainShiftStats verify_main_shift(int W, int K, int ngpu) {
    if (K < 1 || K + 3 > W) fail("main shift geometry");
    const auto main_words = gen_words(W);
    ProductionFactorTables tables(W);
    const OwnerPlan plan = make_owner_plan(W, K, ngpu);
    const int old_start = K + 1; // forward window B=[0,K+1]
    const int new_start = 2;     // reverse window A=[1,K+2]

    std::map<RunKey, MainRunRec> runs;
    for (MateID m : main_words) {
        const Key key{false, m};
        const std::uint32_t support = occupancy_mask(m, W);
        const int occupied = __builtin_popcount(support);
        const Rank pc = catalan((occupied + 1) / 2);
        const Rank pr = tables.primitive_rank(m, W);
        const GroupedRank sr = grouped_rank(
            key, tables, W, 1, false, old_start, K, ngpu, plan);
        const GroupedRank dr = grouped_rank(
            key, tables, W, 2, true, new_start, K, ngpu, plan);
        if (sr.local < pr || dr.local < pr) fail("main shift base underflow");
        const RunKey rk{support, false};
        auto [it, inserted] = runs.emplace(rk, MainRunRec{});
        MainRunRec& z = it->second;
        if (inserted) {
            z.src_owner = sr.owner;
            z.dst_owner = dr.owner;
            z.src_base = sr.local - pr;
            z.dst_base = dr.local - pr;
            z.pc = pc;
        } else if (z.src_owner != sr.owner || z.dst_owner != dr.owner ||
                   z.src_base + pr != sr.local || z.dst_base + pr != dr.local || z.pc != pc) {
            fail("main shift primitive run is not contiguous");
        }
    }

    std::map<std::tuple<int, Rank, Rank>, RunKey> source_interval;
    for (const auto& [rk, z] : runs) {
        if (!source_interval.emplace(std::make_tuple(z.src_owner, z.src_base, z.pc), rk).second)
            fail("main shift source interval collision");
    }

    std::map<RunKey, RunKey> next;
    for (const auto& [rk, z] : runs) {
        const auto it = source_interval.find(std::make_tuple(z.dst_owner, z.dst_base, z.pc));
        if (it == source_interval.end()) fail("main shift destination interval not a main source interval");
        // B=[0,K+1] -> A=[1,K+2].  On the union span [0,K+2], the
        // destination interval of support s is the old interval of s rotated
        // right by one bit.  Reuse the generic shifted-window formula by
        // choosing reverse=true, S=1.
        const std::uint32_t formula = shift_main_support(rk.support, W, K, 1, true);
        if (it->second.blocked || it->second.support != formula)
            fail("main shift support permutation formula");
        next.emplace(rk, RunKey{formula, false});
    }

    MainShiftStats st;
    st.states = main_words.size();
    st.runs = runs.size();
    const int order = K + 3;
    std::set<RunKey> seen;
    for (const auto& [root, z] : runs) {
        (void)z;
        if (seen.count(root)) continue;
        RunKey cur = root;
        int len = 0;
        do {
            if (!seen.insert(cur).second) fail("main shift cycle overlap");
            cur = next.at(cur);
            ++len;
            if (len > order) fail("main shift cycle did not close");
        } while (!(cur.support == root.support && cur.blocked == root.blocked));
        if (order % len != 0) fail("main shift cycle length does not divide K+3");
        ++st.cycles;
        if (len == 1) ++st.fixed_runs;
        st.max_cycle = std::max(st.max_cycle, len);
    }
    if (seen.size() != runs.size()) fail("main shift run coverage");
    return st;
}

Rank main_overlap_weight(int base_occupied, int overlap_bits) {
    Rank states = 0;
    for (int t = 0; t <= overlap_bits; ++t) {
        const int occupied = base_occupied + t;
        if (!(occupied & 1)) continue;
        states += choose_u64(overlap_bits, t) * catalan((occupied + 1) / 2);
    }
    return states;
}

void print_w28_main_shift_traffic() {
    constexpr int W = 28;
    constexpr int K = 12;
    constexpr int L = K + 2;
    constexpr int O = W - L;
    constexpr int overlap_bits = K + 1; // physical bits 1..13
    Rank total = 0, moved = 0;
    Rank pair_states[8][8]{};
    const Rank common_count = Rank(1) << 13; // physical bits 15..27
    for (Rank c = 0; c < common_count; ++c) {
        const int common_ones = __builtin_popcountll(c);
        for (int old_bit = 0; old_bit <= 1; ++old_bit) {
            const std::uint32_t old_outer = (std::uint32_t(c) << 1) | std::uint32_t(old_bit);
            const int so = weighted_owner(old_outer, L, O, 8);
            for (int new_bit = 0; new_bit <= 1; ++new_bit) {
                const std::uint32_t new_outer = (std::uint32_t(c) << 1) | std::uint32_t(new_bit);
                const int d = weighted_owner(new_outer, L, O, 8);
                const Rank states = main_overlap_weight(common_ones + old_bit + new_bit, overlap_bits);
                total += states;
                pair_states[so][d] += states;
                if (so != d) moved += states;
            }
        }
    }
    if (total != 385719506620ULL) fail("W28 main shift total");
    Rank heaviest = 0;
    int hs = -1, hd = -1;
    for (int s = 0; s < 8; ++s) for (int d = 0; d < 8; ++d) {
        if (s != d && pair_states[s][d] > heaviest) {
            heaviest = pair_states[s][d]; hs = s; hd = d;
        }
    }
    std::cout << "W=28 K=12 main_states=" << total
              << " moved_states=" << moved
              << " moved_fraction=" << double(moved) / double(total)
              << " peer_TiB=" << double(moved) * 4.0 / double(1ULL << 40)
              << " heaviest_peer=" << hs << "->" << hd
              << " heaviest_peer_GiB=" << double(heaviest) * 4.0 / double(1ULL << 30)
              << " main_run_count=" << (Rank(1) << 27)
              << " analytic_cycle_order=15"
              << " run_table_bytes=0 visited_bytes=0\n";
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 11;
    const int ngpu = argc > 2 ? std::atoi(argv[2]) : 8;
    if (maxW < 6 || maxW > 12 || ngpu != 8) return 2;

    for (int W = 6; W <= maxW; ++W) {
        for (int K = 2; K <= std::min(4, W - 3); ++K) {
            if (K + 3 > W) continue;
            const MainShiftStats st = verify_main_shift(W, K, ngpu);
            std::cout << "W=" << W
                      << " K=" << K
                      << " states=" << st.states
                      << " primitive_runs=" << st.runs
                      << " cycles=" << st.cycles
                      << " fixed_runs=" << st.fixed_runs
                      << " max_cycle=" << st.max_cycle
                      << " analytic_order=" << (K + 3)
                      << " main_partition_closed=1 in_place=1\n";
        }
    }
    print_w28_main_shift_traffic();
    std::cout << "ALL_OK production_row_turn_main_shift=1\n";
    return 0;
}
