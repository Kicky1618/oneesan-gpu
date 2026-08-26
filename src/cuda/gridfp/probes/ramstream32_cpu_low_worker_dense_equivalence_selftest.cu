#define RAMSTREAM_CPU_LOW_WORKER_LOCALITY_SELFTEST_NO_MAIN
#include "ramstream32_cpu_low_domain_worker_locality_selftest.cu"
#undef RAMSTREAM_CPU_LOW_WORKER_LOCALITY_SELFTEST_NO_MAIN

#include "../ramstream32_cpu_low_domain_worker_unique_dense_coalesce.hpp"

int main() {
    constexpr Count mod = 4294967291u;
    constexpr int W = TARGET_W;
    static_assert(W == LOW_LUT_K + HIGH_LUT_K + 1);
    static_assert(W <= 12, "dense equivalence selftest intentionally uses small W");

    build_full_dp();
    G_FACTOR = build_factor_tables();
    StorageFactorHost storage = build_storage_factor_tables(G_FACTOR);
    StorageLayout layout = build_storage_layout(storage);
    LowDescHost lowdesc = build_low_descriptors(storage, layout);
    LowOrbitHost orbit = build_cpu_low_orbit(storage, layout, lowdesc);
    CpuLowSparseHost sparse = build_cpu_low_sparse(storage, layout, lowdesc, orbit);
    WindowPlan low_wp = make_direct2d_window(false);
    auto jobs = make_cpu_low_jobs(W, low_wp);

    CpuLowSparsePersistentPool flat(4, CPU_LOW_SCHEDULE_DOMAIN, 2, true);
    CpuLowSparsePersistentPool dense(4, CPU_LOW_SCHEDULE_DOMAIN, 2, true);
    flat.prepare_static_schedule(jobs, sparse);
    dense.prepare_static_schedule(jobs, sparse);
    cpu_low_apply_domain_page_tiebreak(flat, jobs, sparse, storage, layout);
    cpu_low_apply_domain_page_tiebreak(dense, jobs, sparse, storage, layout);
    cpu_low_apply_domain_worker_locality(flat, jobs, sparse);
    cpu_low_apply_domain_worker_locality(dense, jobs, sparse);
    if (flat.sticky_worker_jobs != dense.sticky_worker_jobs
        || flat.sticky_worker_cells != dense.sticky_worker_cells) return 1;

    CpuLowDomainWorkerUniqueCoalesceStats flat_stats =
        cpu_low_apply_domain_worker_unique_coalesce(
            flat, jobs, sparse, storage, layout);
    CpuLowDomainWorkerUniqueDenseStats dense_stats =
        cpu_low_apply_domain_worker_unique_dense_coalesce(
            dense, jobs, sparse, storage, layout);

    if (flat.sticky_worker_jobs != dense.sticky_worker_jobs) return 2;
    if (flat.sticky_worker_cells != dense.sticky_worker_cells) return 3;
    if (flat_stats.unique_pages_2m_before != dense_stats.unique_pages_2m_before
        || flat_stats.unique_pages_2m_after != dense_stats.unique_pages_2m_after
        || flat_stats.unique_pages_4k_before != dense_stats.unique_pages_4k_before
        || flat_stats.unique_pages_4k_after != dense_stats.unique_pages_4k_after
        || flat_stats.owner_transitions_before != dense_stats.owner_transitions_before
        || flat_stats.owner_transitions_after != dense_stats.owner_transitions_after)
        return 4;
    if (flat_stats.candidate_evaluations != dense_stats.candidate_evaluations
        || flat_stats.cap_rejections != dense_stats.cap_rejections
        || flat_stats.accepted_moves != dense_stats.accepted_moves
        || flat_stats.unique_page_improving_moves
            != dense_stats.unique_page_improving_moves
        || flat_stats.transition_only_moves != dense_stats.transition_only_moves
        || flat_stats.moved_cells != dense_stats.moved_cells)
        return 5;
    if (flat_stats.flat_delta_normalizations != dense_stats.dense_delta_normalizations)
        return 6;

    auto main_states = enum_states(W);
    auto block_states = enum_states(W - 1);
    if (main_states.size() != layout.main_size
        || block_states.size() != layout.block_size) return 7;
    std::unordered_map<MateID,size_t> mi, di;
    for (size_t i = 0; i < main_states.size(); ++i) mi.emplace(main_states[i], i);
    for (size_t i = 0; i < block_states.size(); ++i) di.emplace(block_states[i], i);

    std::vector<Count> init_main(main_states.size()), init_block(block_states.size());
    std::mt19937_64 rng(301618);
    for (auto& x : init_main) x = Count(rng() % mod);
    for (auto& x : init_block) x = Count(rng() % mod);
    auto one = reference_window(
        W, LOW_LUT_K, 1, mod,
        main_states, block_states, mi, di, init_main, init_block);
    if (one.first.empty()) return 8;
    auto two = reference_window(
        W, LOW_LUT_K, 1, mod,
        main_states, block_states, mi, di, one.first, one.second);
    if (two.first.empty()) return 9;

    RamCounts main_auth, block_auth;
    main_auth.alloc(layout.main_size, "dense equivalence main");
    block_auth.alloc(layout.block_size, "dense equivalence block");
    fill_factor(
        main_auth, block_auth, main_states, block_states,
        init_main, init_block, storage, layout);
    dense.run(jobs, main_auth, block_auth, storage, layout, sparse, mod);
    if (!compare_factor(
            "dense-equivalence-1", main_auth, block_auth,
            main_states, block_states, one.first, one.second,
            storage, layout)) return 10;
    dense.run(jobs, main_auth, block_auth, storage, layout, sparse, mod);
    if (!compare_factor(
            "dense-equivalence-2", main_auth, block_auth,
            main_states, block_states, two.first, two.second,
            storage, layout)) return 11;

    std::cout << "cpu-low-worker-dense-equivalence-selftest OK"
              << " identical_schedule=1 identical_objective=1"
              << " identical_moves=1 exact_generations=2"
              << " flat_build_s=" << flat_stats.build_s
              << " dense_build_s=" << dense_stats.build_s
              << " dense_index_mib="
              << double(dense_stats.dense_index_bytes) / double(1 << 20)
              << '\n';

    flat.shutdown();
    dense.shutdown();
    main_auth.release();
    block_auth.release();
    return 0;
}
