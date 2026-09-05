#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_p2p_tie_balance_probe_main_unused
#include "gridfp_reduced_production_p2p_tie_balance_probe.cpp"
#pragma pop_macro("main")

#include <array>
#include <set>

namespace {

std::uint32_t p2p_batch_hash_cpu(
    std::uint32_t support,
    bool blocked,
    bool reverse
) {
    // Distinct salt from executor tie selection so the two partitions are not
    // artificially correlated, while retaining the same cheap avalanche mix.
    std::uint32_t x = support ^ 0x9e3779b9u;
    x ^= blocked ? 0x27d4eb2du : 0u;
    x ^= reverse ? 0x165667b1u : 0u;
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

struct CycleBatchStats {
    Rank cycles = 0;
    Rank network_cycles = 0;
    Rank local_cycles = 0;
    Rank boundary_runs = 0;
    __uint128_t network_states = 0;
    std::vector<std::array<__uint128_t, 8>> scratch;
    std::array<__uint128_t, 8> inbound{};
    __uint128_t max_cycle_one_gpu = 0;
};

CycleBatchStats verify_cycle_hash_batches(
    int W,
    bool reverse,
    int ngpu,
    int batches
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
            fail("cycle batch run inconsistency");
    }

    CycleBatchStats out;
    out.scratch.resize(static_cast<std::size_t>(batches));
    std::set<RunKey> seen;
    for (const auto& [candidate, rec0] : runs) {
        if (seen.count(candidate)) continue;
        std::vector<RunKey> cycle;
        RunKey cur = candidate;
        do {
            if (!runs.count(cur)) fail("cycle batch escaped run set");
            if (!seen.insert(cur).second) fail("cycle batch overlap");
            cycle.push_back(cur);
            cur = shifted_next_key(cur, W, q, K, K, reverse);
            if (cycle.size() > static_cast<std::size_t>(RP_MAX_W))
                fail("cycle batch cycle too long");
        } while (!(cur.support == candidate.support && cur.blocked == candidate.blocked));

        const Rank pc = rec0.pc;
        std::vector<int> owners;
        owners.reserve(cycle.size());
        for (const RunKey& r : cycle) {
            const auto it = runs.find(r);
            if (it == runs.end() || it->second.pc != pc)
                fail("cycle batch primitive count changed");
            owners.push_back(it->second.owner);
        }

        std::array<Rank, 8> incoming_boundaries{};
        Rank boundaries = 0;
        for (std::size_t i = 0; i < owners.size(); ++i) {
            const std::size_t p = (i + owners.size() - 1) % owners.size();
            if (owners[i] == owners[p]) continue;
            ++boundaries;
            ++incoming_boundaries[static_cast<std::size_t>(owners[i])];
        }
        if (!boundaries) {
            ++out.local_cycles;
            ++out.cycles;
            continue;
        }

        const int batch = int(
            p2p_batch_hash_cpu(candidate.support, candidate.blocked, reverse) %
            std::uint32_t(batches));
        for (int g = 0; g < ngpu; ++g) {
            const __uint128_t z =
                __uint128_t(incoming_boundaries[static_cast<std::size_t>(g)]) * pc;
            out.scratch[static_cast<std::size_t>(batch)][static_cast<std::size_t>(g)] += z;
            out.inbound[static_cast<std::size_t>(g)] += z;
            out.max_cycle_one_gpu = std::max(out.max_cycle_one_gpu, z);
        }
        out.boundary_runs += boundaries;
        out.network_states += __uint128_t(boundaries) * pc;
        ++out.network_cycles;
        ++out.cycles;
    }
    if (seen.size() != runs.size()) fail("cycle batch run coverage");

    __uint128_t scratch_sum = 0;
    for (const auto& batch : out.scratch)
        for (int g = 0; g < ngpu; ++g)
            scratch_sum += batch[static_cast<std::size_t>(g)];
    if (scratch_sum != out.network_states)
        fail("cycle batch scratch/network conservation");
    const ModalTraffic modal = verify_modal_traffic(W, K, reverse, ngpu);
    if (out.network_states != modal.boundary_payload_ops)
        fail("cycle batch exact boundary payload mismatch");
    return out;
}

double u128_to_d(__uint128_t x) {
    return static_cast<double>(x);
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 14;
    const int ngpu = argc > 2 ? std::atoi(argv[2]) : 8;
    const int batches = argc > 3 ? std::atoi(argv[3]) : 8;
    if (maxW < 8 || maxW > 14 || ngpu < 2 || ngpu > 8 || batches < 1 || batches > 64)
        return 2;

    for (int W = 8; W <= maxW; W += 2) {
        for (bool reverse : {false, true}) {
            const CycleBatchStats s = verify_cycle_hash_batches(
                W, reverse, ngpu, batches);
            __uint128_t max_cell = 0;
            double worst_over_avg = 0.0;
            for (int g = 0; g < ngpu; ++g) {
                const double avg = u128_to_d(s.inbound[static_cast<std::size_t>(g)]) /
                                   double(batches);
                for (int b = 0; b < batches; ++b) {
                    const __uint128_t z =
                        s.scratch[static_cast<std::size_t>(b)][static_cast<std::size_t>(g)];
                    max_cell = std::max(max_cell, z);
                    if (avg) worst_over_avg = std::max(worst_over_avg, u128_to_d(z) / avg);
                }
            }
            std::cout << "W=" << W
                      << " K=" << ((W - 2) / 2)
                      << " direction=" << (reverse ? "reverse" : "forward")
                      << " batches=" << batches
                      << " cycles=" << s.cycles
                      << " network_cycles=" << s.network_cycles
                      << " local_cycles=" << s.local_cycles
                      << " boundary_runs=" << s.boundary_runs
                      << " network_states=" << u128_to_d(s.network_states)
                      << " max_batch_gpu_states=" << u128_to_d(max_cell)
                      << " max_batch_gpu_MiB="
                      << u128_to_d(max_cell) * 4.0 / double(1ULL << 20)
                      << " worst_over_destination_average=" << worst_over_avg
                      << " max_single_cycle_gpu_MiB="
                      << u128_to_d(s.max_cycle_one_gpu) * 4.0 / double(1ULL << 20)
                      << " cycle_atomic=1 payload_conserved=1\n";
        }
    }

    std::cout << "W=28_plan initial_batches=8 scratch_cap_GiB_per_gpu=32"
              << " assignment=canonical_cycle_hash"
              << " builder_must_validate_each_batch_gpu_scratch<=cap"
              << " retry_more_batches_or_new_hash_on_overflow=1"
              << " exact_network_payload_TiB=1.490731627"
              << " two_kernels_per_batch=1\n";
    std::cout << "ALL_OK production_p2p_cycle_atomic_hash_batching=1\n";
    return 0;
}
