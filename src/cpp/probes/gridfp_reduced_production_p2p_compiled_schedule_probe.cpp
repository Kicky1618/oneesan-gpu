#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_p2p_worklist_probe_main_unused
#include "gridfp_reduced_production_p2p_worklist_probe.cpp"
#pragma pop_macro("main")

#include <array>
#include <set>

namespace {

static constexpr int COMPILED_LEN_SHIFT = 24;
static constexpr std::uint32_t COMPILED_PC_MASK =
    (std::uint32_t(1) << COMPILED_LEN_SHIFT) - 1u;

struct CompiledCycleHeader {
    std::uint32_t run_begin = 0;
    std::uint32_t primitive_and_len = 0;
};
static_assert(sizeof(CompiledCycleHeader) == 8);

CompiledCycleHeader make_compiled_header(Rank run_begin, Rank pc, int len) {
    if (run_begin > std::numeric_limits<std::uint32_t>::max() ||
        pc > COMPILED_PC_MASK || len < 2 || len > 31)
        fail("compiled header packing range");
    return CompiledCycleHeader{
        static_cast<std::uint32_t>(run_begin),
        static_cast<std::uint32_t>(pc) |
            (static_cast<std::uint32_t>(len) << COMPILED_LEN_SHIFT)};
}

Rank compiled_header_pc(CompiledCycleHeader h) {
    return h.primitive_and_len & COMPILED_PC_MASK;
}

int compiled_header_len(CompiledCycleHeader h) {
    return int(h.primitive_and_len >> COMPILED_LEN_SHIFT);
}

struct CompiledOwnerSchedule {
    std::vector<CompiledCycleHeader> header;
    std::vector<std::uint64_t> run;
};

struct AnalyticRunRank {
    int owner = -1;
    Rank local = 0;
    Rank pc = 0;
};

std::uint32_t local_support_mask(
    std::uint32_t support, int lo, int L
) {
    const std::uint32_t mask = L == 32 ? ~0u : ((std::uint32_t(1) << L) - 1u);
    return (support >> lo) & mask;
}

std::uint32_t erase_two_bits_cpu(
    std::uint32_t local, int L, int a, int b
) {
    std::uint32_t compact = 0;
    int q = 0;
    for (int pos = 0; pos < L; ++pos) {
        if (pos == a || pos == b) continue;
        if ((local >> pos) & 1u) compact |= std::uint32_t(1) << q;
        ++q;
    }
    return compact;
}

AnalyticRunRank analytic_run_rank0(
    RunKey run,
    int W,
    int q,
    bool reverse,
    int tile_start,
    int K,
    int ngpu,
    const OwnerPlan& plan
) {
    const int L = K + 2;
    const int O = W - L;
    const int lo = reverse ? tile_start - 1 : tile_start - K - 1;
    const int hi = lo + L - 1;
    const std::uint32_t outer = compress_outside_window(run.support, W, lo, hi);
    const int outer_ones = __builtin_popcount(outer);
    const Rank group = fixed_outer_group_size(L, outer_ones);
    const Rank sr_outer = support_rank_fixed(outer, O, outer_ones);
    const Rank prefix = group_prefix_before_r(L, O, outer_ones);
    const Rank group_base = prefix + sr_outer * group;
    const int owner = weighted_owner(outer, L, O, ngpu);

    const std::uint32_t local = local_support_mask(run.support, lo, L);
    const int local_ones = __builtin_popcount(local);
    const int occupied = outer_ones + local_ones;
    if (!(occupied & 1)) fail("compiled schedule odd occupancy");
    const Rank pc = catalan((occupied + 1) / 2);
    Rank within = group_local_sector_offset(L, outer_ones, local_ones);
    if (!run.blocked) {
        const Rank sr = rank_bits_mask(local, L, local_ones);
        within += sr * pc;
    } else {
        const int missing_bit = reverse ? q - 1 : q;
        const int fixed_bit = reverse ? q : q - 1;
        if (((run.support >> missing_bit) & 1u) != 0 ||
            ((run.support >> fixed_bit) & 1u) == 0)
            fail("compiled blocked support convention");
        const int missing_pos = missing_bit - lo;
        const int fixed_pos = fixed_bit - lo;
        if (missing_pos < 0 || missing_pos >= L || fixed_pos < 0 || fixed_pos >= L)
            fail("compiled blocked fixed bits outside local window");
        const std::uint32_t compact = erase_two_bits_cpu(
            local, L, missing_pos, fixed_pos);
        const Rank sr = rank_bits_mask(compact, L - 2, local_ones - 1);
        within += choose_u64(L, local_ones) * pc + sr * pc;
    }
    const Rank local_rank = group_base - plan.begin[static_cast<std::size_t>(owner)] + within;
    if (local_rank >= plan.size[static_cast<std::size_t>(owner)])
        fail("compiled analytic local rank range");
    return AnalyticRunRank{owner, local_rank, pc};
}

std::uint64_t pack_compiled_run(int owner, Rank local) {
    if (owner < 0 || owner >= 8 || local >= (Rank(1) << 61))
        fail("compiled run packing range");
    return (std::uint64_t(local) << 3) | std::uint64_t(owner);
}

int compiled_run_owner(std::uint64_t z) {
    return int(z & 7u);
}

Rank compiled_run_local(std::uint64_t z) {
    return Rank(z >> 3);
}

struct ActualRunRec {
    int owner = -1;
    Rank base = 0;
    Rank pc = 0;
};

std::map<RunKey, ActualRunRec> actual_run_map(
    int W, int K, bool reverse, int ngpu
) {
    const int old_start = reverse ? 1 : W - 1;
    const int q = old_start + (reverse ? K : -K);
    ProductionFactorTables tables(W);
    const OwnerPlan plan = make_owner_plan(W, K, ngpu);
    const auto main = gen_words(W);
    const auto block = gen_words(W - 1);
    std::map<RunKey, ActualRunRec> out;
    for (Key key : layout(main, block, q)) {
        const MateID full = embed_full(key, W, q, reverse);
        const std::uint32_t support = occupancy_mask(full, W);
        const Rank pr = tables.primitive_rank(full, W);
        const Rank pc = catalan((__builtin_popcount(support) + 1) / 2);
        const GroupedRank gr = grouped_rank(
            key, tables, W, q, reverse, old_start, K, ngpu, plan);
        if (gr.local < pr) fail("compiled actual run base underflow");
        const RunKey rk{support, key.blocked};
        const ActualRunRec want{gr.owner, gr.local - pr, pc};
        auto [it, inserted] = out.emplace(rk, want);
        if (!inserted &&
            (it->second.owner != want.owner || it->second.base != want.base || it->second.pc != want.pc))
            fail("compiled actual run inconsistency");
    }
    return out;
}

std::array<CompiledOwnerSchedule, 8> compile_schedule(
    int W, int K, bool reverse, int ngpu
) {
    const int old_start = reverse ? 1 : W - 1;
    const int q = old_start + (reverse ? K : -K);
    const Rank base_count = Rank(1) << (W - 2);
    const OwnerPlan plan = make_owner_plan(W, K, ngpu);
    const auto actual = actual_run_map(W, K, reverse, ngpu);

    std::array<CompiledOwnerSchedule, 8> schedule;
    std::set<RunKey> compiled_runs;
    for (Rank base_rank = 0; base_rank < base_count; ++base_rank) {
        RunKey seeds[3]{};
        const int nr = descriptor_run_seeds(base_rank, W, q, reverse, seeds);
        for (int ri = 0; ri < nr; ++ri) {
            const RunKey root = seeds[ri];
            const auto [len, leader] = cycle_length_and_leader(
                root, W, q, K, reverse);
            if (!leader || len <= 1) continue;

            std::array<int, 8> counts{};
            std::array<AnalyticRunRank, RP_MAX_W> rank{};
            RunKey cur = root;
            for (int h = 0; h < len; ++h) {
                if (!compiled_runs.insert(cur).second)
                    fail("compiled schedule duplicate run");
                rank[static_cast<std::size_t>(h)] = analytic_run_rank0(
                    cur, W, q, reverse, old_start, K, ngpu, plan);
                const auto it = actual.find(cur);
                if (it == actual.end()) fail("compiled run outside actual map");
                const auto& a = it->second;
                const auto& r = rank[static_cast<std::size_t>(h)];
                if (a.owner != r.owner || a.base != r.local || a.pc != r.pc)
                    fail("compiled analytic run rank mismatch");
                ++counts[static_cast<std::size_t>(r.owner)];
                cur = shifted_next_key(cur, W, q, K, K, reverse);
            }
            int exec = 0;
            for (int g = 1; g < ngpu; ++g)
                if (counts[static_cast<std::size_t>(g)] >
                    counts[static_cast<std::size_t>(exec)]) exec = g;
            auto& s = schedule[static_cast<std::size_t>(exec)];
            if (s.run.size() > std::numeric_limits<std::uint32_t>::max())
                fail("compiled owner run offset exceeds u32");
            const Rank pc = rank[0].pc;
            for (int h = 1; h < len; ++h)
                if (rank[static_cast<std::size_t>(h)].pc != pc)
                    fail("compiled primitive count changed in cycle");
            s.header.push_back(make_compiled_header(s.run.size(), pc, len));
            for (int h = 0; h < len; ++h) {
                const auto& r = rank[static_cast<std::size_t>(h)];
                s.run.push_back(pack_compiled_run(r.owner, r.local));
            }
        }
    }

    Rank nonfixed_actual = 0;
    for (const auto& [rk, rec] : actual) {
        (void)rec;
        const auto [len, leader] = cycle_length_and_leader(rk, W, q, K, reverse);
        (void)leader;
        if (len > 1) ++nonfixed_actual;
    }
    if (compiled_runs.size() != nonfixed_actual)
        fail("compiled nonfixed run coverage");

    for (int g = 0; g < ngpu; ++g) {
        const auto& s = schedule[static_cast<std::size_t>(g)];
        for (CompiledCycleHeader h : s.header) {
            const Rank begin = h.run_begin;
            const int len = compiled_header_len(h);
            const Rank end = begin + Rank(len);
            const Rank pc = compiled_header_pc(h);
            if (len < 2 || len > RP_MAX_W || !pc || end > s.run.size())
                fail("compiled packed header interval");
            for (Rank j = begin; j < end; ++j) {
                const int owner = compiled_run_owner(s.run[static_cast<std::size_t>(j)]);
                const Rank local = compiled_run_local(s.run[static_cast<std::size_t>(j)]);
                if (owner < 0 || owner >= ngpu ||
                    local >= plan.size[static_cast<std::size_t>(owner)])
                    fail("compiled packed run decode");
            }
        }
    }
    return schedule;
}

void verify_compiled_schedule(int W, bool reverse, int ngpu) {
    const int K = (W - 2) / 2;
    const auto schedule = compile_schedule(W, K, reverse, ngpu);
    Rank cycles = 0, runs = 0;
    Rank bytes = 0;
    for (int g = 0; g < ngpu; ++g) {
        const auto& s = schedule[static_cast<std::size_t>(g)];
        cycles += s.header.size();
        runs += s.run.size();
        bytes += s.header.size() * sizeof(CompiledCycleHeader) +
                 s.run.size() * sizeof(std::uint64_t);
    }
    const WorklistStats w = verify_worklist(W, reverse, ngpu);
    if (cycles != w.descriptors)
        fail("compiled cycle/worklist count mismatch");

    Rank expected_nonfixed_runs = w.runs - w.fixed_cycles;
    if (runs != expected_nonfixed_runs)
        fail("compiled nonfixed run count mismatch");

    std::cout << "W=" << W
              << " K=" << K
              << " direction=" << (reverse ? "reverse" : "forward")
              << " compiled_cycles=" << cycles
              << " compiled_runs=" << runs
              << " compiled_metadata_bytes=" << bytes
              << " run_record_bytes=8 header_bytes=8"
              << " header_order_independent=1"
              << " grouped_rank_per_row=0 support_transform_per_row=0"
              << " owner_compute_per_row=0 exact_run_bases=1\n";
}

void print_w28_compiled_theory() {
    constexpr Rank main_runs = Rank(1) << 27;
    constexpr Rank blocked_runs = Rank(1) << 25;
    constexpr Rank blocked_fixed = Rank(1) << 13;
    constexpr Rank nonfixed_runs = main_runs + blocked_runs - blocked_fixed;
    constexpr Rank cycles = 21566612ULL;
    constexpr Rank run_bytes = nonfixed_runs * sizeof(std::uint64_t);
    constexpr Rank header_bytes = cycles * sizeof(CompiledCycleHeader);
    constexpr Rank total_bytes = run_bytes + header_bytes;
    static_assert(nonfixed_runs == 167763968ULL);
    static_assert(run_bytes == 1342111744ULL);
    static_assert(header_bytes == 172532896ULL);
    static_assert(total_bytes == 1514644640ULL);

    std::cout << "W=28 K=13"
              << " compiled_nonfixed_runs=" << nonfixed_runs
              << " compiled_cycles=" << cycles
              << " run_metadata_MiB=" << double(run_bytes) / double(1ULL << 20)
              << " cycle_header_MiB=" << double(header_bytes) / double(1ULL << 20)
              << " total_compiled_schedule_GiB=" << double(total_bytes) / double(1ULL << 30)
              << " avg_MiB_per_8gpu=" << double(total_bytes) / 8.0 / double(1ULL << 20)
              << " forward_reverse_both_GiB=" << 2.0 * double(total_bytes) / double(1ULL << 30)
              << " header_len_bits=5 primitive_bits=24"
              << " state_stream_GiB_per_gpu=220.442683"
              << " pure_rotation_runtime_candidate=1\n";
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 12;
    const int ngpu = argc > 2 ? std::atoi(argv[2]) : 8;
    if (maxW < 8 || maxW > 14 || ngpu < 2 || ngpu > 8) return 2;

    for (int W = 8; W <= maxW; W += 2)
        for (bool reverse : {false, true})
            verify_compiled_schedule(W, reverse, ngpu);
    print_w28_compiled_theory();
    std::cout << "ALL_OK production_p2p_compiled_schedule=1\n";
    return 0;
}
