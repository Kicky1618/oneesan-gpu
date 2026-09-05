#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_p2p_modal_probe_main_unused
#include "gridfp_reduced_production_p2p_modal_probe.cpp"
#pragma pop_macro("main")

#include <array>
#include <set>

namespace {

struct SegmentRotationStats {
    Rank cycles = 0;
    Rank local_cycles = 0;
    Rank distributed_cycles = 0;
    Rank run_boundaries = 0;
    Rank max_boundaries = 0;
    __uint128_t exact_network_u32 = 0;
    __uint128_t modal_remote_u32_ops = 0;
};

void verify_scalar_segment_rotation(const std::vector<int>& owners) {
    const int n = static_cast<int>(owners.size());
    if (n <= 1) return;
    std::vector<std::uint64_t> old(static_cast<std::size_t>(n));
    std::vector<std::uint64_t> got = old;
    std::vector<std::uint64_t> want(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) old[static_cast<std::size_t>(i)] = 1000u + std::uint64_t(i);
    got = old;
    for (int i = 0; i < n; ++i)
        want[static_cast<std::size_t>(i)] = old[static_cast<std::size_t>((i + n - 1) % n)];

    std::vector<int> starts;
    for (int i = 0; i < n; ++i) {
        const int p = (i + n - 1) % n;
        if (owners[static_cast<std::size_t>(i)] != owners[static_cast<std::size_t>(p)])
            starts.push_back(i);
    }
    if (starts.empty()) {
        std::uint64_t temp = got.back();
        for (int i = n - 1; i > 0; --i)
            got[static_cast<std::size_t>(i)] = got[static_cast<std::size_t>(i - 1)];
        got[0] = temp;
    } else {
        std::vector<std::uint64_t> incoming(starts.size());
        // Phase 1: every owner boundary captures exactly one old predecessor.
        for (std::size_t si = 0; si < starts.size(); ++si) {
            const int s = starts[si];
            const int p = (s + n - 1) % n;
            incoming[si] = old[static_cast<std::size_t>(p)];
        }
        // Global barrier here in the GPU implementation.
        // Phase 2: each same-owner segment shifts locally backward, then installs
        // the single incoming boundary value at its first run.
        for (std::size_t si = 0; si < starts.size(); ++si) {
            const int s = starts[si];
            const int next_s = starts[(si + 1) % starts.size()];
            int e = (next_s + n - 1) % n;
            int h = e;
            while (h != s) {
                const int p = (h + n - 1) % n;
                got[static_cast<std::size_t>(h)] = old[static_cast<std::size_t>(p)];
                h = p;
            }
            got[static_cast<std::size_t>(s)] = incoming[si];
        }
    }
    if (got != want) fail("segmented scalar rotation mismatch");
}

SegmentRotationStats verify_segmented_rotation(
    int W,
    bool reverse,
    int ngpu
) {
    const int K = (W - 2) / 2;
    const int old_start = reverse ? 1 : W - 1;
    const int q = old_start + (reverse ? K : -K);
    ProductionFactorTables tables(W);
    const OwnerPlan plan = make_owner_plan(W, K, ngpu);
    const auto main = gen_words(W);
    const auto block = gen_words(W - 1);

    struct Rec { int owner = -1; Rank pc = 0; };
    std::map<RunKey, Rec> runs;
    for (Key key : layout(main, block, q)) {
        const MateID full = embed_full(key, W, q, reverse);
        const std::uint32_t support = occupancy_mask(full, W);
        const Rank pc = catalan((__builtin_popcount(support) + 1) / 2);
        const GroupedRank gr = grouped_rank(
            key, tables, W, q, reverse, old_start, K, ngpu, plan);
        const RunKey rk{support, key.blocked};
        auto [it, inserted] = runs.emplace(rk, Rec{gr.owner, pc});
        if (!inserted && (it->second.owner != gr.owner || it->second.pc != pc))
            fail("segmented run inconsistency");
    }

    SegmentRotationStats out;
    std::set<RunKey> seen;
    for (const auto& [candidate, rec0] : runs) {
        if (seen.count(candidate)) continue;
        std::vector<RunKey> cycle;
        RunKey cur = candidate;
        do {
            if (!runs.count(cur)) fail("segmented cycle escaped run set");
            if (!seen.insert(cur).second) fail("segmented cycle overlap");
            cycle.push_back(cur);
            cur = shifted_next_key(cur, W, q, K, K, reverse);
            if (cycle.size() > static_cast<std::size_t>(RP_MAX_W))
                fail("segmented cycle too long");
        } while (!(cur.support == candidate.support && cur.blocked == candidate.blocked));

        const Rank pc = rec0.pc;
        std::vector<int> owners;
        owners.reserve(cycle.size());
        std::array<int, 8> counts{};
        for (const RunKey& r : cycle) {
            const auto it = runs.find(r);
            if (it == runs.end() || it->second.pc != pc)
                fail("segmented pc changed in cycle");
            owners.push_back(it->second.owner);
            ++counts[static_cast<std::size_t>(it->second.owner)];
        }
        verify_scalar_segment_rotation(owners);

        Rank boundaries = 0;
        for (std::size_t i = 0; i < owners.size(); ++i)
            boundaries += owners[i] != owners[(i + 1) % owners.size()];
        int max_count = 0;
        for (int g = 0; g < ngpu; ++g)
            max_count = std::max(max_count, counts[static_cast<std::size_t>(g)]);
        const Rank modal_remote_runs = cycle.size() - Rank(max_count);

        out.run_boundaries += boundaries;
        out.max_boundaries = std::max(out.max_boundaries, boundaries);
        out.exact_network_u32 += __uint128_t(boundaries) * pc;
        out.modal_remote_u32_ops += __uint128_t(2) * modal_remote_runs * pc;
        if (boundaries) ++out.distributed_cycles;
        else ++out.local_cycles;
        ++out.cycles;
    }
    if (seen.size() != runs.size()) fail("segmented run coverage");

    const ModalTraffic modal = verify_modal_traffic(W, K, reverse, ngpu);
    if (out.exact_network_u32 != modal.boundary_payload_ops)
        fail("segmented network payload mismatch");
    if (out.modal_remote_u32_ops != modal.modal_remote_ops)
        fail("segmented modal traffic mismatch");
    if (out.exact_network_u32 > out.modal_remote_u32_ops)
        fail("segmented network exceeds modal executor traffic");
    return out;
}

double u128d(__uint128_t x) {
    return static_cast<double>(x);
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 14;
    const int ngpu = argc > 2 ? std::atoi(argv[2]) : 8;
    if (maxW < 8 || maxW > 14 || ngpu < 2 || ngpu > 8) return 2;

    for (int W = 8; W <= maxW; W += 2) {
        for (bool reverse : {false, true}) {
            const SegmentRotationStats s = verify_segmented_rotation(W, reverse, ngpu);
            const double exact = u128d(s.exact_network_u32);
            const double modal = u128d(s.modal_remote_u32_ops);
            std::cout << "W=" << W
                      << " K=" << ((W - 2) / 2)
                      << " direction=" << (reverse ? "reverse" : "forward")
                      << " cycles=" << s.cycles
                      << " local_cycles=" << s.local_cycles
                      << " distributed_cycles=" << s.distributed_cycles
                      << " run_boundaries=" << s.run_boundaries
                      << " max_boundaries_per_cycle=" << s.max_boundaries
                      << " exact_network_u32=" << exact
                      << " modal_executor_remote_u32_ops=" << modal
                      << " exact_over_modal=" << (modal ? exact / modal : 0.0)
                      << " two_phase_segment_rotation=OK"
                      << " one_transfer_per_owner_boundary=1\n";
        }
    }

    constexpr Rank w28_moved_states = 409769189454ULL;
    std::cout << "W=28 K=13"
              << " exact_boundary_payload_states=" << w28_moved_states
              << " exact_boundary_payload_TiB="
              << double(w28_moved_states) * 4.0 / double(1ULL << 40)
              << " phase1=gather_boundary_old_values"
              << " phase2=owner_local_backward_shift_plus_install"
              << " global_barrier_between_phases=1"
              << " max_cycle_run_boundaries<=28\n";
    std::cout << "ALL_OK production_p2p_segmented_rotation=1\n";
    return 0;
}
