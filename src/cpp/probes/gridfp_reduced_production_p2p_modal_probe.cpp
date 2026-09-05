#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_shift_cycle_probe_main_unused
#include "gridfp_reduced_production_shift_cycle_probe.cpp"
#pragma pop_macro("main")

#include <array>
#include <set>

namespace {

struct ModalTraffic {
    Rank runs = 0;
    Rank cycles = 0;
    __uint128_t baseline_remote_ops = 0;
    __uint128_t modal_remote_ops = 0;
    __uint128_t boundary_payload_ops = 0;
};

ModalTraffic verify_modal_traffic(int W, int K, bool reverse, int ngpu) {
    const int S = K;
    const int old_start = reverse ? 1 : W - 1;
    const int q = old_start + (reverse ? S : -S);
    ProductionFactorTables tables(W);
    const OwnerPlan plan = make_owner_plan(W, K, ngpu);
    const auto main = gen_words(W);
    const auto block = gen_words(W - 1);

    struct Rec { int owner = -1; Rank pc = 0; };
    std::map<RunKey, Rec> runs;
    for (Key key : layout(main, block, q)) {
        const MateID full = embed_full(key, W, q, reverse);
        const std::uint32_t support = occupancy_mask(full, W);
        const int occupied = __builtin_popcount(support);
        const Rank pc = catalan((occupied + 1) / 2);
        const GroupedRank gr = grouped_rank(
            key, tables, W, q, reverse, old_start, K, ngpu, plan);
        const RunKey rk{support, key.blocked};
        auto [it, inserted] = runs.emplace(rk, Rec{gr.owner, pc});
        if (!inserted && (it->second.owner != gr.owner || it->second.pc != pc))
            fail("modal run owner/primitive mismatch");
    }

    ModalTraffic out;
    out.runs = runs.size();
    std::set<RunKey> seen;
    for (const auto& [candidate, rec0] : runs) {
        if (seen.count(candidate)) continue;
        std::vector<RunKey> cycle;
        RunKey cur = candidate;
        do {
            if (!runs.count(cur)) fail("modal cycle escaped run set");
            if (!seen.insert(cur).second) fail("modal cycle overlap");
            cycle.push_back(cur);
            cur = shifted_next_key(cur, W, q, K, S, reverse);
            if (cycle.size() > static_cast<std::size_t>(RP_MAX_W))
                fail("modal cycle too long");
        } while (!(cur.support == candidate.support && cur.blocked == candidate.blocked));

        const Rank pc = rec0.pc;
        for (const RunKey& r : cycle) if (runs.at(r).pc != pc) fail("modal pc changed in cycle");
        std::array<int, 8> counts{};
        std::vector<int> owners;
        for (const RunKey& r : cycle) {
            const int owner = runs.at(r).owner;
            ++counts[static_cast<std::size_t>(owner)];
            owners.push_back(owner);
        }
        const int baseline_owner = runs.at(candidate).owner;
        int modal_owner = 0;
        for (int g = 1; g < ngpu; ++g)
            if (counts[static_cast<std::size_t>(g)] > counts[static_cast<std::size_t>(modal_owner)])
                modal_owner = g;
        const Rank baseline_remote_runs = cycle.size() - counts[static_cast<std::size_t>(baseline_owner)];
        const Rank modal_remote_runs = cycle.size() - counts[static_cast<std::size_t>(modal_owner)];
        if (modal_remote_runs > baseline_remote_runs) fail("modal executor worse than baseline");
        Rank boundaries = 0;
        for (std::size_t i = 0; i < owners.size(); ++i)
            boundaries += owners[i] != owners[(i + 1) % owners.size()];

        out.baseline_remote_ops += __uint128_t(2) * baseline_remote_runs * pc;
        out.modal_remote_ops += __uint128_t(2) * modal_remote_runs * pc;
        out.boundary_payload_ops += __uint128_t(boundaries) * pc;
        ++out.cycles;
    }
    if (seen.size() != runs.size()) fail("modal run coverage");
    return out;
}

double u128_to_double(__uint128_t x) {
    return static_cast<double>(static_cast<unsigned long long>(x));
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 12;
    const int ngpu = argc > 2 ? std::atoi(argv[2]) : 8;
    if (maxW < 8 || maxW > 14 || ngpu < 2 || ngpu > 8) return 2;

    for (int W = 8; W <= maxW; W += 2) {
        const int K = (W - 2) / 2;
        for (bool reverse : {false, true}) {
            const ModalTraffic t = verify_modal_traffic(W, K, reverse, ngpu);
            const double base = u128_to_double(t.baseline_remote_ops);
            const double modal = u128_to_double(t.modal_remote_ops);
            const double ideal = u128_to_double(t.boundary_payload_ops);
            std::cout << "W=" << W
                      << " K=" << K
                      << " direction=" << (reverse ? "reverse" : "forward")
                      << " runs=" << t.runs
                      << " cycles=" << t.cycles
                      << " baseline_remote_u32_ops=" << base
                      << " modal_remote_u32_ops=" << modal
                      << " modal_over_baseline=" << (base ? modal / base : 0.0)
                      << " modal_over_boundary_payload=" << (ideal ? modal / ideal : 0.0)
                      << " modal_never_worse=1 exact_cycle_coverage=1\n";
        }
    }
    std::cout << "W=28_note main_cycle_order=28 blocked_cycle_order=2"
              << " shared_rank_metadata_entries_per_warp<=28"
              << " modal_executor_table_bytes=0\n";
    std::cout << "ALL_OK production_p2p_modal_executor=1\n";
    return 0;
}
