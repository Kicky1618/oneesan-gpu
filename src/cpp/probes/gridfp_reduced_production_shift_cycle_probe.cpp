#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_tile_run_probe_main_unused
#include "gridfp_reduced_production_tile_run_probe.cpp"
#pragma pop_macro("main")

#include <numeric>

namespace {

std::uint32_t rotate_bits(std::uint32_t x, int len, int shift) {
    if (len <= 0) fail("shift rotate length");
    shift %= len;
    if (shift < 0) shift += len;
    if (!shift) return x;
    const std::uint32_t mask = (std::uint32_t(1) << len) - 1u;
    return ((x << shift) | (x >> (len - shift))) & mask;
}

std::uint32_t shift_main_support(
    std::uint32_t support,
    int W,
    int Kwin,
    int S,
    bool reverse
) {
    const int span = Kwin + S + 2;
    const int lo = reverse ? 0 : W - span;
    const std::uint32_t mask = ((std::uint32_t(1) << span) - 1u) << lo;
    const std::uint32_t x = (support & mask) >> lo;
    const int shift = reverse ? span - S : S;
    return (support & ~mask) | (rotate_bits(x, span, shift) << lo);
}

std::uint32_t shift_blocked_support(
    std::uint32_t support,
    int W,
    int q,
    int Kwin,
    int S,
    bool reverse
) {
    const int span = Kwin + S + 2;
    const int compact_len = span - 2;
    const int lo = reverse ? 0 : W - span;
    std::uint32_t compact = 0;
    int cp = 0;
    for (int bit = lo; bit < lo + span; ++bit) {
        if (bit == q - 1 || bit == q) continue;
        if ((support >> bit) & 1u) compact |= std::uint32_t(1) << cp;
        ++cp;
    }
    if (cp != compact_len) fail("blocked compact span length");
    const int shift = reverse ? compact_len - S : S;
    const std::uint32_t rotated = rotate_bits(compact, compact_len, shift);
    std::uint32_t out = support;
    cp = 0;
    for (int bit = lo; bit < lo + span; ++bit) {
        if (bit == q - 1 || bit == q) continue;
        out &= ~(std::uint32_t(1) << bit);
        if ((rotated >> cp) & 1u) out |= std::uint32_t(1) << bit;
        ++cp;
    }
    return out;
}

RunKey shifted_next_key(
    RunKey k,
    int W,
    int q,
    int Kwin,
    int S,
    bool reverse
) {
    k.support = k.blocked
        ? shift_blocked_support(k.support, W, q, Kwin, S, reverse)
        : shift_main_support(k.support, W, Kwin, S, reverse);
    return k;
}

struct ShiftRunRec {
    int src_owner = -1;
    int dst_owner = -1;
    Rank src_base = 0;
    Rank dst_base = 0;
    Rank pc = 0;
};

struct ShiftCycleStats {
    Rank runs = 0;
    Rank fixed_runs = 0;
    Rank nontrivial_cycles = 0;
    int max_main_cycle = 0;
    int max_blocked_cycle = 0;
};

ShiftCycleStats verify_shift_cycles(
    const std::vector<MateID>& main,
    const std::vector<MateID>& block,
    int W,
    int Kwin,
    int S,
    bool reverse,
    int ngpu
) {
    if (Kwin < 1 || S < 1 || S > Kwin || Kwin + S + 2 > W)
        fail("shift cycle geometry");
    const int old_start = reverse ? 1 : W - 1;
    const int q = old_start + (reverse ? S : -S);
    const int new_start = q;
    ProductionFactorTables tables(W);
    const OwnerPlan plan = make_owner_plan(W, Kwin, ngpu);

    std::map<RunKey, ShiftRunRec> runs;
    for (Key key : layout(main, block, q)) {
        const MateID full = embed_full(key, W, q, reverse);
        const std::uint32_t support = occupancy_mask(full, W);
        const int occupied = __builtin_popcount(support);
        const Rank pc = catalan((occupied + 1) / 2);
        const Rank pr = tables.primitive_rank(full, W);
        const GroupedRank sr = grouped_rank(
            key, tables, W, q, reverse, old_start, Kwin, ngpu, plan);
        const GroupedRank dr = grouped_rank(
            key, tables, W, q, reverse, new_start, Kwin, ngpu, plan);
        if (sr.local < pr || dr.local < pr) fail("shift run base underflow");

        const RunKey rk{support, key.blocked};
        auto [it, inserted] = runs.emplace(rk, ShiftRunRec{});
        ShiftRunRec& z = it->second;
        if (inserted) {
            z.src_owner = sr.owner;
            z.dst_owner = dr.owner;
            z.src_base = sr.local - pr;
            z.dst_base = dr.local - pr;
            z.pc = pc;
        } else if (z.src_owner != sr.owner || z.dst_owner != dr.owner ||
                   z.src_base + pr != sr.local || z.dst_base + pr != dr.local || z.pc != pc) {
            fail("shift primitive run is not contiguous");
        }
    }

    std::map<std::tuple<int, Rank, Rank>, RunKey> source_interval;
    for (const auto& [rk, z] : runs) {
        if (!source_interval.emplace(std::make_tuple(z.src_owner, z.src_base, z.pc), rk).second)
            fail("shift source interval collision");
    }

    std::map<RunKey, RunKey> next;
    for (const auto& [rk, z] : runs) {
        const auto it = source_interval.find(std::make_tuple(z.dst_owner, z.dst_base, z.pc));
        if (it == source_interval.end()) fail("shift destination interval not in source partition");
        const RunKey formula = shifted_next_key(rk, W, q, Kwin, S, reverse);
        if (it->second.support != formula.support || it->second.blocked != formula.blocked)
            fail(std::string(reverse ? "reverse" : "forward") +
                 " shifted support permutation formula W=" + std::to_string(W) +
                 " Kwin=" + std::to_string(Kwin) + " S=" + std::to_string(S));
        next.emplace(rk, formula);
    }

    ShiftCycleStats st;
    st.runs = runs.size();
    std::set<RunKey> seen;
    const int main_order = (Kwin + S + 2) / std::gcd(Kwin + S + 2, S);
    const int blocked_order = (Kwin + S) / std::gcd(Kwin + S, S);
    for (const auto& [root, z] : runs) {
        (void)z;
        if (seen.count(root)) continue;
        RunKey cur = root;
        int len = 0;
        do {
            if (!seen.insert(cur).second) fail("shift cycle overlap");
            cur = next.at(cur);
            ++len;
            const int bound = root.blocked ? blocked_order : main_order;
            if (len > bound) fail("shift cycle did not close within analytic order");
        } while (cur.support != root.support || cur.blocked != root.blocked);
        if (len == 1) ++st.fixed_runs;
        else ++st.nontrivial_cycles;
        if (root.blocked) {
            if (blocked_order % len != 0) fail("blocked shift cycle order");
            st.max_blocked_cycle = std::max(st.max_blocked_cycle, len);
        } else {
            if (main_order % len != 0) fail("main shift cycle order");
            st.max_main_cycle = std::max(st.max_main_cycle, len);
        }
    }
    if (seen.size() != runs.size()) fail("shift cycle coverage");
    return st;
}

struct GenericOverlapWeight {
    Rank states = 0;
    Rank runs = 0;
};

GenericOverlapWeight generic_overlap_weight(int base_occupied, int overlap_bits) {
    if (overlap_bits < 2 || overlap_bits > 20) fail("overlap width");
    GenericOverlapWeight out;
    const int fixed = overlap_bits - 2;
    const int missing = overlap_bits - 1;
    for (std::uint32_t mask = 0; mask < (std::uint32_t(1) << overlap_bits); ++mask) {
        const int occupied = base_occupied + __builtin_popcount(mask);
        if (!(occupied & 1)) continue;
        const Rank pc = catalan((occupied + 1) / 2);
        out.states += pc;
        ++out.runs;
        if (((mask >> fixed) & 1u) && !((mask >> missing) & 1u)) {
            out.states += pc;
            ++out.runs;
        }
    }
    return out;
}

RunTraffic shifted_run_traffic_model(int W, int Kwin, int S, int ngpu) {
    if (ngpu != 8) fail("shift traffic currently fixed to 8 GPUs");
    const int L = Kwin + 2;
    const int common_bits = W - (L + S);
    const int overlap_bits = L - S;
    if (common_bits < 0 || overlap_bits < 2) fail("shift traffic geometry");

    RunTraffic out;
    const Rank common_count = Rank(1) << common_bits;
    for (Rank c = 0; c < common_count; ++c) {
        const RunOwnerHist old_hist = run_owner_hist_for_common(
            static_cast<std::uint32_t>(c), common_bits, S, L, ngpu);
        const RunOwnerHist new_hist = run_owner_hist_for_common(
            static_cast<std::uint32_t>(c), common_bits, S, L, ngpu);
        const int common_ones = __builtin_popcountll(c);
        for (int so = 0; so < ngpu; ++so) {
            for (int bo = 0; bo <= S; ++bo) {
                const Rank cb = old_hist.by_owner_ones[static_cast<std::size_t>(so)]
                                                      [static_cast<std::size_t>(bo)];
                if (!cb) continue;
                for (int d = 0; d < ngpu; ++d) {
                    for (int ao = 0; ao <= S; ++ao) {
                        const Rank ca = new_hist.by_owner_ones[static_cast<std::size_t>(d)]
                                                          [static_cast<std::size_t>(ao)];
                        if (!ca) continue;
                        const GenericOverlapWeight w = generic_overlap_weight(
                            common_ones + bo + ao, overlap_bits);
                        const Rank pairs = cb * ca;
                        const Rank states = pairs * w.states;
                        const Rank runs = pairs * w.runs;
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
    if (out.total_runs != expected_runs) fail("shift traffic total runs");
    return out;
}

void print_w28_shift_plan(int Kwin, int S) {
    const RunTraffic t = shifted_run_traffic_model(28, Kwin, S, 8);
    if (t.total_states != 473397057701ULL) fail("W28 shift traffic state total");
    const int main_order = (Kwin + S + 2) / std::gcd(Kwin + S + 2, S);
    const int blocked_order = (Kwin + S) / std::gcd(Kwin + S, S);
    std::cout << "W=28 Kwin=" << Kwin
              << " shift=" << S
              << " overlap_bits=" << (Kwin + 2 - S)
              << " moved_states=" << t.moved_states
              << " moved_fraction=" << double(t.moved_states) / double(t.total_states)
              << " redistribution_TiB=" << double(t.moved_states) * 4.0 / double(1ULL << 40)
              << " moved_runs=" << t.moved_runs
              << " main_cycle_order=" << main_order
              << " blocked_cycle_order=" << blocked_order
              << " run_table_bytes=0 visited_bytes=0 scratch_bytes=0\n";
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 11;
    const int ngpu = argc > 2 ? std::atoi(argv[2]) : 8;
    if (maxW < 7 || maxW > 12 || ngpu != 8) return 2;

    std::vector<std::vector<MateID>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 7; W <= maxW; ++W) {
        for (int Kwin = 2; Kwin <= std::min(4, W - 3); ++Kwin) {
            for (int S = 1; S <= Kwin && Kwin + S + 2 <= W; ++S) {
                for (bool reverse : {false, true}) {
                    const ShiftCycleStats s = verify_shift_cycles(
                        words[W], words[W - 1], W, Kwin, S, reverse, ngpu);
                    std::cout << "W=" << W
                              << " Kwin=" << Kwin
                              << " shift=" << S
                              << " direction=" << (reverse ? "reverse" : "forward")
                              << " runs=" << s.runs
                              << " max_main_cycle=" << s.max_main_cycle
                              << " max_blocked_cycle=" << s.max_blocked_cycle
                              << " primitive_partition_preserved=1"
                              << " main_rotation_formula=1"
                              << " blocked_compact_rotation_formula=1"
                              << " OK\n";
                }
            }
        }
    }

    const RunTraffic k12 = shifted_run_traffic_model(28, 12, 12, 8);
    const RunTraffic k13 = shifted_run_traffic_model(28, 13, 12, 8);
    if (k12.moved_states != 387344269008ULL || k12.moved_runs != 106060788ULL)
        fail("W28 K12 shift constants");
    if (k13.moved_states != 396783615236ULL || k13.moved_runs != 114145920ULL)
        fail("W28 K13 shift constants");
    print_w28_shift_plan(12, 12);
    print_w28_shift_plan(13, 12);
    std::cout << "ALL_OK production_shifted_equal_window_cycles=1\n";
    return 0;
}
