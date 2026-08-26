#define RAMSTREAM_CPU_LOW_WORKER_LOCALITY_SELFTEST_NO_MAIN
#include "ramstream32_cpu_low_domain_worker_locality_selftest.cu"
#undef RAMSTREAM_CPU_LOW_WORKER_LOCALITY_SELFTEST_NO_MAIN

#include "../ramstream32_cpu_low_domain_worker_multistart.hpp"
#include "../ramstream32_cpu_low_domain_worker_unique_shared_coalesce.hpp"

static bool shared_test_same(
    const CpuLowSparsePersistentPool& a,
    const CpuLowSparsePersistentPool& b
) {
    return a.sticky_worker_jobs == b.sticky_worker_jobs
        && a.sticky_worker_cells == b.sticky_worker_cells;
}

int main() {
    constexpr Count mod = 4294967291u;
    constexpr int W = TARGET_W;
    static_assert(W == LOW_LUT_K + HIGH_LUT_K + 1);
    static_assert(W <= 12, "shared multistart selftest intentionally uses small W");

    build_full_dp();
    G_FACTOR = build_factor_tables();
    StorageFactorHost storage = build_storage_factor_tables(G_FACTOR);
    StorageLayout layout = build_storage_layout(storage);
    LowDescHost lowdesc = build_low_descriptors(storage, layout);
    LowOrbitHost orbit = build_cpu_low_orbit(storage, layout, lowdesc);
    CpuLowSparseHost sparse = build_cpu_low_sparse(storage, layout, lowdesc, orbit);
    WindowPlan low_wp = make_direct2d_window(false);
    auto jobs = make_cpu_low_jobs(W, low_wp);

    CpuLowSparsePersistentPool ld(4, CPU_LOW_SCHEDULE_DOMAIN, 2, true);
    CpuLowSparsePersistentPool lh(4, CPU_LOW_SCHEDULE_DOMAIN, 2, true);
    CpuLowSparsePersistentPool sd(4, CPU_LOW_SCHEDULE_DOMAIN, 2, true);
    CpuLowSparsePersistentPool sh(4, CPU_LOW_SCHEDULE_DOMAIN, 2, true);
    CpuLowSparsePersistentPool selected(4, CPU_LOW_SCHEDULE_DOMAIN, 2, true);
    for (CpuLowSparsePersistentPool* p : {&ld, &lh, &sd, &sh, &selected}) {
        p->prepare_static_schedule(jobs, sparse);
        cpu_low_apply_domain_page_tiebreak(*p, jobs, sparse, storage, layout);
        cpu_low_apply_domain_worker_locality(*p, jobs, sparse);
    }
    if (!shared_test_same(ld, lh) || !shared_test_same(ld, sd)
        || !shared_test_same(ld, sh) || !shared_test_same(ld, selected)) return 1;

    CpuLowWorkerExactWorkspace ws = cpu_low_build_worker_exact_workspace(
        jobs, sparse, storage, layout);
    CpuLowDomainWorkerUniqueCoalesceStats ld_stats =
        cpu_low_apply_domain_worker_unique_coalesce(ld, jobs, sparse, storage, layout);
    CpuLowDomainWorkerCoalesceStats lh_local =
        cpu_low_apply_domain_worker_coalesce(lh, jobs, sparse, storage, layout);
    CpuLowDomainWorkerUniqueCoalesceStats lh_stats =
        cpu_low_apply_domain_worker_unique_coalesce(lh, jobs, sparse, storage, layout);

    CpuLowDomainWorkerUniqueDenseStats sd_stats =
        cpu_low_apply_domain_worker_unique_shared_coalesce(sd, jobs, sparse, ws);
    CpuLowDomainWorkerCoalesceStats sh_local =
        cpu_low_apply_domain_worker_coalesce(sh, jobs, sparse, storage, layout);
    CpuLowDomainWorkerUniqueDenseStats sh_stats =
        cpu_low_apply_domain_worker_unique_shared_coalesce(sh, jobs, sparse, ws);

    if (!shared_test_same(ld, sd) || !shared_test_same(lh, sh)) return 2;
    if (ld_stats.candidate_evaluations != sd_stats.candidate_evaluations
        || ld_stats.accepted_moves != sd_stats.accepted_moves
        || lh_stats.candidate_evaluations != sh_stats.candidate_evaluations
        || lh_stats.accepted_moves != sh_stats.accepted_moves
        || lh_local.accepted_moves != sh_local.accepted_moves) return 3;

    CpuLowDomainWorkerUniqueCoalesceStats compat_sd{};
    compat_sd.unique_pages_2m_after = sd_stats.unique_pages_2m_after;
    compat_sd.unique_pages_4k_after = sd_stats.unique_pages_4k_after;
    compat_sd.owner_transitions_after = sd_stats.owner_transitions_after;
    CpuLowDomainWorkerUniqueCoalesceStats compat_sh{};
    compat_sh.unique_pages_2m_after = sh_stats.unique_pages_2m_after;
    compat_sh.unique_pages_4k_after = sh_stats.unique_pages_4k_after;
    compat_sh.owner_transitions_after = sh_stats.owner_transitions_after;
    CpuLowDomainWorkerMultiStartDecision legacy_decision =
        cpu_low_choose_worker_multistart(ld, ld_stats, lh, lh_stats);
    CpuLowDomainWorkerMultiStartDecision shared_decision =
        cpu_low_choose_worker_multistart(sd, compat_sd, sh, compat_sh);
    if (legacy_decision.source != shared_decision.source) return 4;

    const CpuLowSparsePersistentPool& winner =
        shared_decision.source == CPU_LOW_WORKER_MULTISTART_HYBRID ? sh : sd;
    cpu_low_copy_worker_static_schedule(selected, winner);
    if (!shared_test_same(selected, winner)) return 5;

    auto main_states = enum_states(W);
    auto block_states = enum_states(W - 1);
    if (main_states.size() != layout.main_size
        || block_states.size() != layout.block_size) return 6;
    std::unordered_map<MateID,size_t> mi, di;
    for (size_t i = 0; i < main_states.size(); ++i) mi.emplace(main_states[i], i);
    for (size_t i = 0; i < block_states.size(); ++i) di.emplace(block_states[i], i);
    std::vector<Count> init_main(main_states.size()), init_block(block_states.size());
    std::mt19937_64 rng(311618);
    for (auto& x : init_main) x = Count(rng() % mod);
    for (auto& x : init_block) x = Count(rng() % mod);
    auto one = reference_window(
        W, LOW_LUT_K, 1, mod,
        main_states, block_states, mi, di, init_main, init_block);
    if (one.first.empty()) return 7;
    auto two = reference_window(
        W, LOW_LUT_K, 1, mod,
        main_states, block_states, mi, di, one.first, one.second);
    if (two.first.empty()) return 8;

    RamCounts main_auth, block_auth;
    main_auth.alloc(layout.main_size, "shared multistart main");
    block_auth.alloc(layout.block_size, "shared multistart block");
    fill_factor(
        main_auth, block_auth, main_states, block_states,
        init_main, init_block, storage, layout);
    selected.run(jobs, main_auth, block_auth, storage, layout, sparse, mod);
    if (!compare_factor(
            "shared-multistart-1", main_auth, block_auth,
            main_states, block_states, one.first, one.second,
            storage, layout)) return 9;
    selected.run(jobs, main_auth, block_auth, storage, layout, sparse, mod);
    if (!compare_factor(
            "shared-multistart-2", main_auth, block_auth,
            main_states, block_states, two.first, two.second,
            storage, layout)) return 10;

    std::cout << "cpu-low-worker-shared-multistart-selftest OK"
              << " selected_source="
              << cpu_low_worker_multistart_source_name(shared_decision.source)
              << " identical_direct_schedule=1 identical_hybrid_schedule=1"
              << " identical_selector=1 exact_generations=2"
              << " workspace_mib=" << double(ws.bytes()) / double(1 << 20)
              << " workspace_build_s=" << ws.build_s
              << '\n';

    ld.shutdown(); lh.shutdown(); sd.shutdown(); sh.shutdown(); selected.shutdown();
    main_auth.release(); block_auth.release();
    return 0;
}
