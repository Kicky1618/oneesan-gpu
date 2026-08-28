#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_grouped_codec_probe_main_unused
#include "gridfp_reduced_production_grouped_codec_probe.cpp"
#pragma pop_macro("main")

#include <map>
#include <set>

namespace {

struct SlabKey {
    bool blocked = false;
    std::uint32_t support = 0;
    bool operator<(const SlabKey& o) const {
        return blocked != o.blocked ? blocked < o.blocked : support < o.support;
    }
};

struct Slab {
    Rank old_base = 0;
    Rank new_base = 0;
    Rank values = 0;
    std::vector<std::uint8_t> primitive_seen;
};

Rank grouped_global_rank(const GroupedRank& r, const OwnerPlan& plan) {
    if (r.owner < 0 || r.owner >= static_cast<int>(plan.begin.size()))
        fail("slab grouped owner range");
    return plan.begin[static_cast<std::size_t>(r.owner)] + r.local;
}

struct CycleStats {
    Rank states = 0;
    Rank slabs = 0;
    Rank cycles = 0;
    Rank fixed = 0;
    Rank nontrivial = 0;
    Rank max_cycle = 0;
    Rank max_slab_values = 0;
    Rank cross_owner_cycles = 0;
};

CycleStats verify_slab_redistribution(int W, bool reverse, int ngpu) {
    if ((W & 1) || W < 8) fail("slab probe expects even W>=8");
    const int K = W / 2 - 1;
    const int q = W / 2;
    const int old_start = reverse ? 1 : W - 1;
    const int new_start = q;
    const int L = K + 2;
    const int old_lo = reverse ? old_start - 1 : old_start - K - 1;
    const int old_hi = reverse ? old_start + K : old_start;
    const int new_lo = reverse ? new_start - 1 : new_start - K - 1;
    const int new_hi = reverse ? new_start + K : new_start;
    if (old_hi - old_lo + 1 != L || new_hi - new_lo + 1 != L)
        fail("slab window width");
    if (!reverse) {
        if (old_lo != W / 2 - 1 || old_hi != W - 1 || new_lo != 0 || new_hi != W / 2)
            fail("slab forward symmetric windows");
    } else {
        if (old_lo != 0 || old_hi != W / 2 || new_lo != W / 2 - 1 || new_hi != W - 1)
            fail("slab reverse symmetric windows");
    }

    ProductionFactorTables tables(W);
    const OwnerPlan old_plan = make_owner_plan(W, K, ngpu);
    const OwnerPlan new_plan = make_owner_plan(W, K, ngpu);
    if (old_plan.size != new_plan.size || old_plan.begin != new_plan.begin)
        fail("slab owner plans differ for equal window width");

    const auto states = layout(gen_words(W), gen_words(W - 1), q);
    if (states.size() != tables.size()) fail("slab layout dimension");

    std::map<SlabKey, Slab> slabs;
    std::vector<Rank> expected_old_to_new(static_cast<std::size_t>(tables.size()));
    std::vector<std::uint8_t> old_rank_seen(static_cast<std::size_t>(tables.size()));
    std::vector<std::uint8_t> new_rank_seen(static_cast<std::size_t>(tables.size()));

    for (Key k : states) {
        const MateID old_full = embed_full(k, W, q, reverse);
        const MateID new_full = embed_full(k, W, q, reverse);
        if (old_full != new_full) fail("slab full mate changed across layouts");
        const std::uint32_t support = occupancy_mask(old_full, W);
        const int occupied = __builtin_popcount(support);
        if (!(occupied & 1)) fail("slab occupied parity");
        const Rank pc = catalan((occupied + 1) / 2);
        const Rank pr = tables.primitive_rank(old_full, W);
        if (pr >= pc) fail("slab primitive rank range");

        const GroupedRank old_r = grouped_rank(
            k, tables, W, q, reverse, old_start, K, ngpu, old_plan);
        const GroupedRank new_r = grouped_rank(
            k, tables, W, q, reverse, new_start, K, ngpu, new_plan);
        const Rank old_global = grouped_global_rank(old_r, old_plan);
        const Rank new_global = grouped_global_rank(new_r, new_plan);
        if (old_global >= tables.size() || new_global >= tables.size())
            fail("slab global grouped rank range");
        if (old_rank_seen[static_cast<std::size_t>(old_global)]++ ||
            new_rank_seen[static_cast<std::size_t>(new_global)]++)
            fail("slab grouped rank collision");
        expected_old_to_new[static_cast<std::size_t>(old_global)] = new_global;

        const SlabKey key{k.blocked, support};
        const Rank old_base = old_global - pr;
        const Rank new_base = new_global - pr;
        auto [it, inserted] = slabs.emplace(key, Slab{});
        Slab& s = it->second;
        if (inserted) {
            s.old_base = old_base;
            s.new_base = new_base;
            s.values = pc;
            s.primitive_seen.assign(static_cast<std::size_t>(pc), 0);
        } else if (s.old_base != old_base || s.new_base != new_base || s.values != pc) {
            fail("slab base/size depends on primitive rank");
        }
        if (s.primitive_seen[static_cast<std::size_t>(pr)]++)
            fail("slab duplicate primitive rank");
    }

    for (std::uint8_t x : old_rank_seen) if (x != 1) fail("slab old rank hole");
    for (std::uint8_t x : new_rank_seen) if (x != 1) fail("slab new rank hole");

    std::map<Rank, SlabKey> old_base_to_key;
    std::map<Rank, Rank> old_base_size;
    Rank main_slabs = 0;
    Rank blocked_slabs = 0;
    CycleStats st;
    st.states = tables.size();
    st.slabs = slabs.size();
    for (const auto& [key, s] : slabs) {
        for (std::uint8_t x : s.primitive_seen) if (x != 1) fail("slab primitive rank hole");
        if (!old_base_to_key.emplace(s.old_base, key).second)
            fail("slab duplicate old base");
        old_base_size.emplace(s.old_base, s.values);
        st.max_slab_values = std::max(st.max_slab_values, s.values);
        if (key.blocked) ++blocked_slabs;
        else ++main_slabs;
    }

    // Every destination slab must land exactly on one old-layout slab boundary,
    // and the old slab occupying that physical interval must have the same
    // primitive count. This is the condition that upgrades a state permutation
    // to a size-compatible block permutation.
    std::map<Rank, Rank> next;
    for (const auto& [key, s] : slabs) {
        (void)key;
        const auto it = old_base_size.find(s.new_base);
        if (it == old_base_size.end()) fail("slab destination is not old slab boundary");
        if (it->second != s.values) fail("slab destination old slab has different size");
        if (!next.emplace(s.old_base, s.new_base).second)
            fail("slab duplicate permutation source");
    }
    if (next.size() != slabs.size()) fail("slab permutation size");

    // Physical owner of a global grouped slot. OwnerPlan groups are monotone
    // contiguous intervals; use them to identify P2P-crossing cycles.
    auto owner_of = [&](Rank global) {
        for (int g = 0; g < ngpu; ++g) {
            const Rank b = old_plan.begin[static_cast<std::size_t>(g)];
            const Rank e = b + old_plan.size[static_cast<std::size_t>(g)];
            if (global >= b && global < e) return g;
        }
        fail("slab physical owner lookup");
        return -1;
    };

    std::set<Rank> visited;
    std::vector<Rank> cycle;
    for (const auto& [start, ignored] : next) {
        (void)ignored;
        if (visited.count(start)) continue;
        cycle.clear();
        Rank cur = start;
        while (!visited.count(cur)) {
            visited.insert(cur);
            cycle.push_back(cur);
            const auto it = next.find(cur);
            if (it == next.end()) fail("slab cycle escapes permutation");
            cur = it->second;
        }
        if (cur != start) fail("slab permutation merged into previous cycle");
        ++st.cycles;
        st.max_cycle = std::max<Rank>(st.max_cycle, cycle.size());
        if (cycle.size() == 1) ++st.fixed;
        else ++st.nontrivial;
        const Rank bytes = old_base_size.at(start);
        std::set<int> owners;
        for (Rank b : cycle) {
            if (old_base_size.at(b) != bytes) fail("slab cycle mixes block sizes");
            owners.insert(owner_of(b));
        }
        if (owners.size() > 1) ++st.cross_owner_cycles;
    }
    if (visited.size() != slabs.size()) fail("slab cycle coverage");

    // Materialize a small test vector and perform the block-cycle rotation in
    // place. For a cycle b0->b1->...->bk->b0, save the old last block, shift
    // predecessors backwards, then restore the saved last block at b0. Only one
    // maximum-size slab scratch is ever live.
    std::vector<std::uint64_t> initial(static_cast<std::size_t>(tables.size()));
    std::vector<std::uint64_t> expected(static_cast<std::size_t>(tables.size()));
    std::vector<std::uint64_t> work(static_cast<std::size_t>(tables.size()));
    for (Rank r = 0; r < tables.size(); ++r) {
        initial[static_cast<std::size_t>(r)] =
            0x9e3779b97f4a7c15ULL ^ (r * 0xbf58476d1ce4e5b9ULL);
        work[static_cast<std::size_t>(r)] = initial[static_cast<std::size_t>(r)];
        expected[static_cast<std::size_t>(expected_old_to_new[static_cast<std::size_t>(r)])] =
            initial[static_cast<std::size_t>(r)];
    }

    visited.clear();
    std::vector<std::uint64_t> scratch;
    for (const auto& [start, ignored] : next) {
        (void)ignored;
        if (visited.count(start)) continue;
        cycle.clear();
        Rank cur = start;
        while (!visited.count(cur)) {
            visited.insert(cur);
            cycle.push_back(cur);
            cur = next.at(cur);
        }
        if (cycle.size() == 1) continue;
        const Rank n = old_base_size.at(cycle.front());
        scratch.assign(
            work.begin() + static_cast<std::ptrdiff_t>(cycle.back()),
            work.begin() + static_cast<std::ptrdiff_t>(cycle.back() + n));
        for (std::size_t i = cycle.size() - 1; i > 0; --i) {
            const Rank from = cycle[i - 1];
            const Rank to = cycle[i];
            std::copy_n(
                work.begin() + static_cast<std::ptrdiff_t>(from),
                static_cast<std::size_t>(n),
                work.begin() + static_cast<std::ptrdiff_t>(to));
        }
        std::copy(
            scratch.begin(), scratch.end(),
            work.begin() + static_cast<std::ptrdiff_t>(cycle.front()));
    }
    if (work != expected) fail("slab in-place cycle rotation mismatch");

    const Rank want_main = Rank(1) << (W - 1);
    const Rank want_blocked = Rank(1) << (W - 3);
    if (main_slabs != want_main || blocked_slabs != want_blocked)
        fail("slab count closed form");

    std::cout << "W=" << W
              << " direction=" << (reverse ? "reverse" : "forward")
              << " q=" << q
              << " K=" << K
              << " left_window=0.." << W / 2
              << " right_window=" << W / 2 - 1 << ".." << W - 1
              << " states=" << st.states
              << " slabs=" << st.slabs
              << " cycles=" << st.cycles
              << " fixed=" << st.fixed
              << " nontrivial=" << st.nontrivial
              << " max_cycle=" << st.max_cycle
              << " max_slab_values=" << st.max_slab_values
              << " cross_owner_cycles=" << st.cross_owner_cycles
              << " primitive_order_preserved=1"
              << " destination_slab_boundary_exact=1"
              << " one_slab_scratch=1"
              << " in_place_rotation=OK\n";
    return st;
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 12;
    const int ngpu = argc > 2 ? std::atoi(argv[2]) : 8;
    if (maxW < 8 || maxW > 12 || ngpu < 2 || ngpu > 16) return 2;

    for (int W = 8; W <= maxW; W += 2) {
        const CycleStats f = verify_slab_redistribution(W, false, ngpu);
        const CycleStats r = verify_slab_redistribution(W, true, ngpu);
        if (f.states != r.states || f.slabs != r.slabs || f.cycles != r.cycles ||
            f.fixed != r.fixed || f.nontrivial != r.nontrivial ||
            f.max_cycle != r.max_cycle || f.max_slab_values != r.max_slab_values)
            fail("slab forward/reverse reflection stats differ");
    }

    constexpr Rank w28_main_slabs = Rank(1) << 27;
    constexpr Rank w28_blocked_slabs = Rank(1) << 25;
    constexpr Rank w28_slabs = w28_main_slabs + w28_blocked_slabs;
    constexpr Rank w28_max_slab_values = 2674440ULL; // Catalan(14)
    constexpr Rank w28_scratch_bytes = w28_max_slab_values * 4ULL;
    constexpr Rank w28_visited_bytes = (w28_slabs + 7ULL) / 8ULL;
    std::cout << "W=28_theory q=14 K=13"
              << " left_window=0..14 right_window=13..27"
              << " main_slabs=" << w28_main_slabs
              << " blocked_slabs=" << w28_blocked_slabs
              << " total_slabs=" << w28_slabs
              << " max_slab_values=" << w28_max_slab_values
              << " max_u32_scratch_bytes=" << w28_scratch_bytes
              << " max_u32_scratch_MiB=" << double(w28_scratch_bytes) / double(1ULL << 20)
              << " visited_bitset_bytes=" << w28_visited_bytes
              << " visited_bitset_MiB=" << double(w28_visited_bytes) / double(1ULL << 20)
              << " full_second_stream_bytes=0"
              << " redistributions_per_row=1"
              << " row_turn_redistributions=0\n";
    std::cout << "ALL_OK production_support_slab_cycle_redistribution=1\n";
    return 0;
}
