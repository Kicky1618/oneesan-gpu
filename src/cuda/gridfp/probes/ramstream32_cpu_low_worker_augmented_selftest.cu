#define RAMSTREAM_CPU_LOW_WORKER_LOCALITY_SELFTEST_NO_MAIN
#include "ramstream32_cpu_low_domain_worker_locality_selftest.cu"
#undef RAMSTREAM_CPU_LOW_WORKER_LOCALITY_SELFTEST_NO_MAIN

#include "../ramstream32_cpu_low_domain_worker_multistart.hpp"
#include "../ramstream32_cpu_low_domain_worker_unique_shared_coalesce.hpp"
#include "../ramstream32_cpu_low_domain_worker_augmented_fixedpoint.hpp"

static uint64_t aug_self_max(const CpuLowSparsePersistentPool& p) {
    return p.sticky_worker_cells.empty() ? 0
        : *std::max_element(p.sticky_worker_cells.begin(), p.sticky_worker_cells.end());
}

int main() {
    constexpr Count mod = 4294967291u;
    constexpr int W = TARGET_W;
    static_assert(W == LOW_LUT_K + HIGH_LUT_K + 1);
    static_assert(W <= 12, "augmented selftest uses small W");

    build_full_dp();
    G_FACTOR = build_factor_tables();
    StorageFactorHost storage = build_storage_factor_tables(G_FACTOR);
    StorageLayout layout = build_storage_layout(storage);
    LowDescHost lowdesc = build_low_descriptors(storage, layout);
    LowOrbitHost orbit = build_cpu_low_orbit(storage, layout, lowdesc);
    CpuLowSparseHost sparse = build_cpu_low_sparse(storage, layout, lowdesc, orbit);
    WindowPlan low_wp = make_direct2d_window(false);
    auto jobs = make_cpu_low_jobs(W, low_wp);

    CpuLowSparsePersistentPool direct(4, CPU_LOW_SCHEDULE_DOMAIN, 2, true);
    CpuLowSparsePersistentPool hybrid(4, CPU_LOW_SCHEDULE_DOMAIN, 2, true);
    CpuLowSparsePersistentPool parent(4, CPU_LOW_SCHEDULE_DOMAIN, 2, true);
    for (auto* p : {&direct, &hybrid, &parent}) {
        p->prepare_static_schedule(jobs, sparse);
        cpu_low_apply_domain_page_tiebreak(*p, jobs, sparse, storage, layout);
        cpu_low_apply_domain_worker_locality(*p, jobs, sparse);
    }
    CpuLowWorkerExactWorkspace ws =
        cpu_low_build_worker_exact_workspace(jobs, sparse, storage, layout);

    auto ds = cpu_low_apply_domain_worker_unique_shared_coalesce(
        direct, jobs, sparse, ws);
    cpu_low_apply_domain_worker_coalesce(hybrid, jobs, sparse, storage, layout);
    auto hs = cpu_low_apply_domain_worker_unique_shared_coalesce(
        hybrid, jobs, sparse, ws);
    CpuLowDomainWorkerUniqueCoalesceStats cd{}, ch{};
    cd.unique_pages_2m_after = ds.unique_pages_2m_after;
    cd.unique_pages_4k_after = ds.unique_pages_4k_after;
    cd.owner_transitions_after = ds.owner_transitions_after;
    ch.unique_pages_2m_after = hs.unique_pages_2m_after;
    ch.unique_pages_4k_after = hs.unique_pages_4k_after;
    ch.owner_transitions_after = hs.owner_transitions_after;
    auto md = cpu_low_choose_worker_multistart(direct, cd, hybrid, ch);
    cpu_low_copy_worker_static_schedule(
        parent, md.source == CPU_LOW_WORKER_MULTISTART_HYBRID ? hybrid : direct);

    CpuLowSparsePersistentPool baseline(4, CPU_LOW_SCHEDULE_DOMAIN, 2, true);
    CpuLowSparsePersistentPool augmented(4, CPU_LOW_SCHEDULE_DOMAIN, 2, true);
    baseline.prepare_static_schedule(jobs, sparse);
    augmented.prepare_static_schedule(jobs, sparse);
    cpu_low_copy_worker_static_schedule(baseline, parent);
    cpu_low_copy_worker_static_schedule(augmented, parent);

    auto base = cpu_low_apply_best_exact_fixedpoint(
        baseline, jobs, sparse, ws, 4, 4);
    auto az = cpu_low_apply_worker_augmented_fixedpoint(
        augmented, jobs, sparse, ws, 4, 4);
    if (az.neutral_limit_hit || az.round_limit_hit) return 1;
    if (cpu_low_worker_unique_score_less(base.after, az.after)) return 2;
    if (aug_self_max(augmented) > aug_self_max(baseline)) return 3;
    auto bp = cpu_low_worker_sorted_load_profile(baseline.sticky_worker_cells);
    auto ap = cpu_low_worker_sorted_load_profile(augmented.sticky_worker_cells);
    if (cpu_low_worker_score_equal_aug(base.after, az.after) && bp < ap) return 4;

    auto main_states = enum_states(W);
    auto block_states = enum_states(W - 1);
    if (main_states.size() != layout.main_size
        || block_states.size() != layout.block_size) return 5;
    std::unordered_map<MateID,size_t> mi, di;
    for (size_t i = 0; i < main_states.size(); ++i) mi.emplace(main_states[i], i);
    for (size_t i = 0; i < block_states.size(); ++i) di.emplace(block_states[i], i);
    std::vector<Count> init_main(main_states.size()), init_block(block_states.size());
    std::mt19937_64 rng(361618);
    for (auto& x : init_main) x = Count(rng() % mod);
    for (auto& x : init_block) x = Count(rng() % mod);

    auto one = reference_window(
        W, LOW_LUT_K, 1, mod,
        main_states, block_states, mi, di, init_main, init_block);
    if (one.first.empty()) return 6;
    auto two = reference_window(
        W, LOW_LUT_K, 1, mod,
        main_states, block_states, mi, di, one.first, one.second);
    if (two.first.empty()) return 7;

    RamCounts main_auth, block_auth;
    main_auth.alloc(layout.main_size, "augmented selftest main");
    block_auth.alloc(layout.block_size, "augmented selftest block");
    fill_factor(
        main_auth, block_auth, main_states, block_states,
        init_main, init_block, storage, layout);
    augmented.run(jobs, main_auth, block_auth, storage, layout, sparse, mod);
    if (!compare_factor(
            "augmented-1", main_auth, block_auth,
            main_states, block_states, one.first, one.second,
            storage, layout)) return 8;
    augmented.run(jobs, main_auth, block_auth, storage, layout, sparse, mod);
    if (!compare_factor(
            "augmented-2", main_auth, block_auth,
            main_states, block_states, two.first, two.second,
            storage, layout)) return 9;

    std::cout << "cpu-low-worker-augmented-selftest OK"
              << " multistart_source="
              << cpu_low_worker_multistart_source_name(md.source)
              << " baseline_2m=" << base.after.pages_2m
              << " augmented_2m=" << az.after.pages_2m
              << " baseline_4k=" << base.after.pages_4k
              << " augmented_4k=" << az.after.pages_4k
              << " baseline_transitions=" << base.after.transitions
              << " augmented_transitions=" << az.after.transitions
              << " rounds=" << az.rounds
              << " neutral_moves=" << az.neutral_moves
              << " exact_generations=2\n";

    direct.shutdown();
    hybrid.shutdown();
    parent.shutdown();
    baseline.shutdown();
    augmented.shutdown();
    main_auth.release();
    block_auth.release();
    return 0;
}
