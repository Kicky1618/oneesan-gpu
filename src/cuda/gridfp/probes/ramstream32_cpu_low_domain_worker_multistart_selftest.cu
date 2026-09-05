// Reuse the independent W10 recurrence/reference helpers from the existing
// worker-locality exactness test without duplicating them. Its dedicated
// NO_MAIN guard exposes the helpers without macro-renaming main across headers.
#define RAMSTREAM_CPU_LOW_WORKER_LOCALITY_SELFTEST_NO_MAIN
#include "ramstream32_cpu_low_domain_worker_locality_selftest.cu"
#undef RAMSTREAM_CPU_LOW_WORKER_LOCALITY_SELFTEST_NO_MAIN

#include "../ramstream32_cpu_low_domain_worker_multistart.hpp"

static uint64_t multistart_max_cells(const CpuLowSparsePersistentPool& pool) {
    return pool.sticky_worker_cells.empty() ? 0
        : *std::max_element(
            pool.sticky_worker_cells.begin(), pool.sticky_worker_cells.end());
}

int main() {
    constexpr Count mod = 4294967291u;
    constexpr int W = TARGET_W;
    static_assert(W == LOW_LUT_K + HIGH_LUT_K + 1);
    static_assert(W <= 12, "multistart selftest intentionally uses small W");

    build_full_dp();
    G_FACTOR = build_factor_tables();
    StorageFactorHost storage = build_storage_factor_tables(G_FACTOR);
    StorageLayout layout = build_storage_layout(storage);
    LowDescHost lowdesc = build_low_descriptors(storage, layout);
    LowOrbitHost orbit = build_cpu_low_orbit(storage, layout, lowdesc);
    CpuLowSparseHost sparse = build_cpu_low_sparse(storage, layout, lowdesc, orbit);
    WindowPlan low_wp = make_direct2d_window(false);
    auto jobs = make_cpu_low_jobs(W, low_wp);

    auto main_states = enum_states(W);
    auto block_states = enum_states(W - 1);
    if (main_states.size() != layout.main_size
        || block_states.size() != layout.block_size) return 2;

    std::unordered_map<MateID,size_t> mi, di;
    mi.reserve(main_states.size() * 2);
    di.reserve(block_states.size() * 2);
    for (size_t i = 0; i < main_states.size(); ++i) mi.emplace(main_states[i], i);
    for (size_t i = 0; i < block_states.size(); ++i) di.emplace(block_states[i], i);

    std::vector<Count> init_main(main_states.size()), init_block(block_states.size());
    std::mt19937_64 rng(281618);
    for (auto& x : init_main) x = Count(rng() % mod);
    for (auto& x : init_block) x = Count(rng() % mod);

    auto one = reference_window(
        W, LOW_LUT_K, 1, mod,
        main_states, block_states, mi, di, init_main, init_block);
    if (one.first.empty()) return 3;
    auto two = reference_window(
        W, LOW_LUT_K, 1, mod,
        main_states, block_states, mi, di, one.first, one.second);
    if (two.first.empty()) return 4;

    CpuLowSparsePersistentPool direct(4, CPU_LOW_SCHEDULE_DOMAIN, 2, true);
    CpuLowSparsePersistentPool hybrid(4, CPU_LOW_SCHEDULE_DOMAIN, 2, true);
    CpuLowSparsePersistentPool selected(4, CPU_LOW_SCHEDULE_DOMAIN, 2, true);
    direct.prepare_static_schedule(jobs, sparse);
    hybrid.prepare_static_schedule(jobs, sparse);
    selected.prepare_static_schedule(jobs, sparse);

    cpu_low_apply_domain_page_tiebreak(direct, jobs, sparse, storage, layout);
    cpu_low_apply_domain_page_tiebreak(hybrid, jobs, sparse, storage, layout);
    cpu_low_apply_domain_page_tiebreak(selected, jobs, sparse, storage, layout);
    cpu_low_apply_domain_worker_locality(direct, jobs, sparse);
    cpu_low_apply_domain_worker_locality(hybrid, jobs, sparse);
    cpu_low_apply_domain_worker_locality(selected, jobs, sparse);

    if (direct.sticky_worker_jobs != hybrid.sticky_worker_jobs
        || direct.sticky_worker_cells != hybrid.sticky_worker_cells
        || direct.sticky_worker_jobs != selected.sticky_worker_jobs
        || direct.sticky_worker_cells != selected.sticky_worker_cells) {
        std::cerr << "multistart selftest v5.25 branch mismatch\n";
        return 5;
    }
    uint64_t parent_max = multistart_max_cells(direct);

    CpuLowDomainWorkerUniqueCoalesceStats direct_stats =
        cpu_low_apply_domain_worker_unique_coalesce(
            direct, jobs, sparse, storage, layout);
    CpuLowDomainWorkerCoalesceStats raw_stats =
        cpu_low_apply_domain_worker_coalesce(
            hybrid, jobs, sparse, storage, layout);
    CpuLowDomainWorkerUniqueScore raw_score{
        0, 0, 0
    };
    CpuLowDomainWorkerUniqueCoalesceStats hybrid_stats =
        cpu_low_apply_domain_worker_unique_coalesce(
            hybrid, jobs, sparse, storage, layout);
    raw_score = {
        hybrid_stats.unique_pages_2m_before,
        hybrid_stats.unique_pages_4k_before,
        hybrid_stats.owner_transitions_before
    };

    CpuLowDomainWorkerMultiStartDecision decision = cpu_low_choose_worker_multistart(
        direct, direct_stats, hybrid, hybrid_stats);
    const CpuLowSparsePersistentPool& winner =
        decision.source == CPU_LOW_WORKER_MULTISTART_HYBRID ? hybrid : direct;
    cpu_low_copy_worker_static_schedule(selected, winner);
    if (selected.sticky_worker_jobs != winner.sticky_worker_jobs
        || selected.sticky_worker_cells != winner.sticky_worker_cells) return 6;

    CpuLowDomainWorkerUniqueScore direct_score =
        cpu_low_worker_unique_after_score(direct_stats);
    CpuLowDomainWorkerUniqueScore hybrid_score =
        cpu_low_worker_unique_after_score(hybrid_stats);
    CpuLowDomainWorkerUniqueScore selected_score =
        decision.source == CPU_LOW_WORKER_MULTISTART_HYBRID
        ? hybrid_score : direct_score;
    CpuLowDomainWorkerUniqueScore parent_score{
        direct_stats.unique_pages_2m_before,
        direct_stats.unique_pages_4k_before,
        direct_stats.owner_transitions_before
    };

    if (cpu_low_worker_unique_score_less(parent_score, direct_score)) return 7;
    if (cpu_low_worker_unique_score_less(raw_score, hybrid_score)) return 8;
    if (cpu_low_worker_unique_score_less(direct_score, selected_score)
        || cpu_low_worker_unique_score_less(hybrid_score, selected_score)
        || cpu_low_worker_unique_score_less(raw_score, selected_score)
        || cpu_low_worker_unique_score_less(parent_score, selected_score)) return 9;
    if (multistart_max_cells(selected) > parent_max
        || raw_stats.max_worker_cells_after > parent_max) return 10;

    RamCounts main_auth, block_auth;
    main_auth.alloc(layout.main_size, "multistart selftest main");
    block_auth.alloc(layout.block_size, "multistart selftest block");
    fill_factor(
        main_auth, block_auth, main_states, block_states,
        init_main, init_block, storage, layout);

    selected.run(jobs, main_auth, block_auth, storage, layout, sparse, mod);
    if (!compare_factor(
            "domain-worker-multistart-1", main_auth, block_auth,
            main_states, block_states, one.first, one.second,
            storage, layout)) return 11;
    selected.run(jobs, main_auth, block_auth, storage, layout, sparse, mod);
    if (!compare_factor(
            "domain-worker-multistart-2", main_auth, block_auth,
            main_states, block_states, two.first, two.second,
            storage, layout)) return 12;

    std::cout << "cpu-low-domain-worker-multistart-selftest OK"
              << " selected_source="
              << cpu_low_worker_multistart_source_name(decision.source)
              << " parent_pages_2m=" << parent_score.pages_2m
              << " selected_pages_2m=" << selected_score.pages_2m
              << " parent_pages_4k=" << parent_score.pages_4k
              << " selected_pages_4k=" << selected_score.pages_4k
              << " parent_transitions=" << parent_score.transitions
              << " selected_transitions=" << selected_score.transitions
              << " parent_max=" << parent_max
              << " selected_max=" << multistart_max_cells(selected)
              << " direct_moves=" << direct_stats.accepted_moves
              << " hybrid_v526_moves=" << raw_stats.accepted_moves
              << " hybrid_v527_moves=" << hybrid_stats.accepted_moves
              << " exact_generations=2\n";

    direct.shutdown();
    hybrid.shutdown();
    selected.shutdown();
    main_auth.release();
    block_auth.release();
    return 0;
}
