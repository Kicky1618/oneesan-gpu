#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#define RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../oneesan_cuda_gridfp_ramstream32_factorized_bidesc_compact.cu"
#undef RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../ramstream32_cpu_low_domain_page.hpp"
#include "../ramstream32_cpu_high.hpp"
#include "../ramstream32_cpu_high_direct_persistent.hpp"

static void enum_states_rec(int pos, int h, MateID m, std::vector<MateID>& out) {
    if (pos < 0) { if (h == 0) out.push_back(m); return; }
    enum_states_rec(pos - 1, h, mset(m, pos, N), out);
    if (h > 0) enum_states_rec(pos - 1, h - 1, mset(m, pos, R), out);
    enum_states_rec(pos - 1, h + 1, mset(m, pos, ::L), out);
}
static std::vector<MateID> enum_states(int width) {
    std::vector<MateID> out; enum_states_rec(width - 1, 1, 0, out); return out;
}
static inline Count ref_add(Count a, Count b, Count mod) {
    return (a >= mod - b) ? a - (mod - b) : a + b;
}
static void fill_factor(
    RamCounts& main_auth, RamCounts& block_auth,
    const std::vector<MateID>& main_states, const std::vector<MateID>& block_states,
    const std::vector<Count>& main_values, const std::vector<Count>& block_values,
    const StorageFactorHost& storage, const StorageLayout& layout
) {
    std::memset(main_auth.ptr, 0, main_auth.bytes);
    std::memset(block_auth.ptr, 0, block_auth.bytes);
    for (size_t i = 0; i < main_states.size(); ++i)
        main_auth.ptr[storage_rank_main_host(main_states[i], storage, layout)] = main_values[i];
    for (size_t i = 0; i < block_states.size(); ++i)
        block_auth.ptr[storage_rank_block_host(block_states[i], storage, layout)] = block_values[i];
}
static bool compare_factor(
    const char* tag, const RamCounts& main_auth, const RamCounts& block_auth,
    const std::vector<MateID>& main_states, const std::vector<MateID>& block_states,
    const std::vector<Count>& rm, const std::vector<Count>& rd,
    const StorageFactorHost& storage, const StorageLayout& layout
) {
    for (size_t i = 0; i < main_states.size(); ++i) {
        Count got = main_auth.ptr[storage_rank_main_host(main_states[i], storage, layout)];
        if (got != rm[i]) { std::cerr << "FAIL " << tag << " main i=" << i << " got=" << got << " want=" << rm[i] << '\n'; return false; }
    }
    for (size_t i = 0; i < block_states.size(); ++i) {
        Count got = block_auth.ptr[storage_rank_block_host(block_states[i], storage, layout)];
        if (got != rd[i]) { std::cerr << "FAIL " << tag << " block i=" << i << " got=" << got << " want=" << rd[i] << '\n'; return false; }
    }
    return true;
}

static std::pair<std::vector<Count>, std::vector<Count>> reference_window(
    int W, int p_hi, int p_lo, Count mod,
    const std::vector<MateID>& main_states,
    const std::vector<MateID>& block_states,
    const std::unordered_map<MateID, size_t>& mi,
    const std::unordered_map<MateID, size_t>& di,
    const std::vector<Count>& init_m,
    const std::vector<Count>& init_d
) {
    std::vector<Count> rm = init_m, rd = init_d;
    for (int p = p_hi; p >= p_lo; --p) {
        std::vector<Count> nm = rm;
        std::vector<Count> nd(rd.size(), 0);
        for (size_t i = 0; i < main_states.size(); ++i) {
            Count c = rm[i];
            auto z = oneesan::gridfp::include_horizontal(main_states[i], W, p);
            if (!z.valid) continue;
            if (z.blocked) {
                auto it = di.find(z.mate); if (it == di.end()) std::exit(3);
                nd[it->second] = ref_add(nd[it->second], c, mod);
            } else {
                auto it = mi.find(z.mate); if (it == mi.end()) std::exit(4);
                nm[it->second] = ref_add(nm[it->second], c, mod);
            }
        }
        for (size_t i = 0; i < block_states.size(); ++i) {
            Count c = rd[i];
            MateID z = oneesan::gridfp::blocked_exclude(block_states[i], p);
            auto it = mi.find(z); if (it == mi.end()) std::exit(5);
            nm[it->second] = ref_add(nm[it->second], c, mod);
        }
        rm.swap(nm); rd.swap(nd);
    }
    return {std::move(rm), std::move(rd)};
}

int main() {
    constexpr Count mod = 4294967291u;
    constexpr int W = TARGET_W;
    static_assert(W == LOW_LUT_K + HIGH_LUT_K + 1);
    static_assert(W <= 12, "selftest intentionally uses a small width");

    build_full_dp();
    G_FACTOR = build_factor_tables();
    StorageFactorHost storage = build_storage_factor_tables(G_FACTOR);
    StorageLayout layout = build_storage_layout(storage);
    LowDescHost lowdesc = build_low_descriptors(storage, layout);
    HighDescHost highdesc = build_high_descriptors(storage, layout);
    LowOrbitHost orbit = build_cpu_low_orbit(storage, layout, lowdesc);
    CpuLowSparseHost sparse = build_cpu_low_sparse(storage, layout, lowdesc, orbit);
    CpuHighCrossHost highcross = build_cpu_high_cross(storage);
    CpuHighDirectHost highdirect = build_cpu_high_direct(storage, layout, highdesc);

    auto main_states = enum_states(W);
    auto block_states = enum_states(W - 1);
    if (main_states.size() != layout.main_size || block_states.size() != layout.block_size) return 2;

    std::unordered_map<MateID, size_t> mi, di;
    mi.reserve(main_states.size() * 2); di.reserve(block_states.size() * 2);
    for (size_t i = 0; i < main_states.size(); ++i) mi.emplace(main_states[i], i);
    for (size_t i = 0; i < block_states.size(); ++i) di.emplace(block_states[i], i);

    RamCounts main_auth, block_auth;
    main_auth.alloc(layout.main_size, "selftest main");
    block_auth.alloc(layout.block_size, "selftest block");
    std::vector<Count> init_m(main_states.size()), init_d(block_states.size());
    std::mt19937_64 rng(1618);
    for (auto& v : init_m) v = Count(rng() % mod);
    for (auto& v : init_d) v = Count(rng() % mod);

    auto [low_rm, low_rd] = reference_window(
        W, LOW_LUT_K, 1, mod, main_states, block_states, mi, di, init_m, init_d);
    auto [low2_rm, low2_rd] = reference_window(
        W, LOW_LUT_K, 1, mod, main_states, block_states, mi, di, low_rm, low_rd);
    WindowPlan low_wp = make_direct2d_window(false);
    auto low_jobs = make_cpu_low_jobs(W, low_wp);

    uint64_t refine_before = 0, refine_after = 0, refine_moves = 0;
    int refine_boundaries = 0;
    {
        if (low_jobs.size() < 6) return 28;
        std::vector<CpuLowJob> refine_jobs(low_jobs.begin(), low_jobs.begin() + 6);
        const uint64_t weights[6] = {8, 8, 8, 1, 1, 1};
        std::vector<CpuLowStaticJobCost> ordered;
        for (size_t i = 0; i < 6; ++i)
            ordered.push_back({i, uint32_t(i), weights[i]});
        std::vector<std::pair<size_t,size_t>> segs = {{0, 3}, {3, 6}};
        refine_before = std::max(
            cpu_low_lpt_range_max_cells(ordered, 0, 3, 1, refine_jobs),
            cpu_low_lpt_range_max_cells(ordered, 3, 6, 1, refine_jobs));
        cpu_low_refine_domain_boundaries(
            ordered, refine_jobs, 2, 1, segs,
            refine_boundaries, refine_moves);
        if (segs[0].second != segs[1].first) return 29;
        refine_after = std::max(
            cpu_low_lpt_range_max_cells(ordered, segs[0].first, segs[0].second, 1, refine_jobs),
            cpu_low_lpt_range_max_cells(ordered, segs[1].first, segs[1].second, 1, refine_jobs));
        if (refine_after > refine_before || refine_boundaries <= 0 || refine_moves == 0)
            return 30;
    }

    fill_factor(main_auth, block_auth, main_states, block_states, init_m, init_d, storage, layout);
    CpuLowPool out_pool(2);
    out_pool.run(low_jobs, main_auth, block_auth, storage, layout, lowdesc, mod);
    if (!compare_factor("out-of-place", main_auth, block_auth, main_states, block_states, low_rm, low_rd, storage, layout)) return 10;

    fill_factor(main_auth, block_auth, main_states, block_states, init_m, init_d, storage, layout);
    CpuLowInplacePool in_pool(2);
    in_pool.run(low_jobs, main_auth, block_auth, storage, layout, lowdesc, orbit, mod);
    if (!compare_factor("in-place", main_auth, block_auth, main_states, block_states, low_rm, low_rd, storage, layout)) return 11;

    fill_factor(main_auth, block_auth, main_states, block_states, init_m, init_d, storage, layout);
    CpuLowDirectPool direct_pool(2);
    direct_pool.run(low_jobs, main_auth, block_auth, storage, layout, lowdesc, orbit, mod);
    if (!compare_factor("direct", main_auth, block_auth, main_states, block_states, low_rm, low_rd, storage, layout)) return 12;

    fill_factor(main_auth, block_auth, main_states, block_states, init_m, init_d, storage, layout);
    CpuLowSparsePool sparse_pool(2);
    sparse_pool.run(low_jobs, main_auth, block_auth, storage, layout, sparse, mod);
    if (!compare_factor("sparse-v4.7", main_auth, block_auth, main_states, block_states, low_rm, low_rd, storage, layout)) return 13;

    fill_factor(main_auth, block_auth, main_states, block_states, init_m, init_d, storage, layout);
    CpuLowSparsePersistentPool sparse_persistent_pool(2, CPU_LOW_SCHEDULE_DYNAMIC);
    sparse_persistent_pool.run(low_jobs, main_auth, block_auth, storage, layout, sparse, mod);
    if (!compare_factor("sparse-persistent-1", main_auth, block_auth,
                        main_states, block_states, low_rm, low_rd, storage, layout)) return 20;
    sparse_persistent_pool.run(low_jobs, main_auth, block_auth, storage, layout, sparse, mod);
    if (!compare_factor("sparse-persistent-2", main_auth, block_auth,
                        main_states, block_states, low2_rm, low2_rd, storage, layout)) return 21;

    fill_factor(main_auth, block_auth, main_states, block_states, init_m, init_d, storage, layout);
    CpuLowSparsePersistentPool sparse_sticky_pool(2, CPU_LOW_SCHEDULE_STICKY);
    sparse_sticky_pool.run(low_jobs, main_auth, block_auth, storage, layout, sparse, mod);
    if (!compare_factor("sparse-sticky-1", main_auth, block_auth,
                        main_states, block_states, low_rm, low_rd, storage, layout)) return 22;
    sparse_sticky_pool.run(low_jobs, main_auth, block_auth, storage, layout, sparse, mod);
    if (!compare_factor("sparse-sticky-2", main_auth, block_auth,
                        main_states, block_states, low2_rm, low2_rd, storage, layout)) return 23;

    fill_factor(main_auth, block_auth, main_states, block_states, init_m, init_d, storage, layout);
    CpuLowSparsePersistentPool sparse_contiguous_pool(2, CPU_LOW_SCHEDULE_CONTIGUOUS);
    sparse_contiguous_pool.run(low_jobs, main_auth, block_auth, storage, layout, sparse, mod);
    if (!compare_factor("sparse-contiguous-1", main_auth, block_auth,
                        main_states, block_states, low_rm, low_rd, storage, layout)) return 24;
    sparse_contiguous_pool.run(low_jobs, main_auth, block_auth, storage, layout, sparse, mod);
    if (!compare_factor("sparse-contiguous-2", main_auth, block_auth,
                        main_states, block_states, low2_rm, low2_rd, storage, layout)) return 25;

    fill_factor(main_auth, block_auth, main_states, block_states, init_m, init_d, storage, layout);
    CpuLowSparsePersistentPool sparse_domain_pool(2, CPU_LOW_SCHEDULE_DOMAIN, 1, true);
    sparse_domain_pool.run(low_jobs, main_auth, block_auth, storage, layout, sparse, mod);
    if (!compare_factor("sparse-domain-1", main_auth, block_auth,
                        main_states, block_states, low_rm, low_rd, storage, layout)) return 26;
    sparse_domain_pool.run(low_jobs, main_auth, block_auth, storage, layout, sparse, mod);
    if (!compare_factor("sparse-domain-2", main_auth, block_auth,
                        main_states, block_states, low2_rm, low2_rd, storage, layout)) return 27;

    fill_factor(main_auth, block_auth, main_states, block_states, init_m, init_d, storage, layout);
    CpuLowSparsePersistentPool sparse_domain_page_pool(2, CPU_LOW_SCHEDULE_DOMAIN, 1, true);
    sparse_domain_page_pool.prepare_static_schedule(low_jobs, sparse);
    CpuLowDomainPageTieStats page_tie_stats = cpu_low_apply_domain_page_tiebreak(
        sparse_domain_page_pool, low_jobs, sparse, storage, layout);
    sparse_domain_page_pool.schedule_build_s += page_tie_stats.build_s;
    if (page_tie_stats.max_worker_cells_after > page_tie_stats.max_worker_cells_before
        || page_tie_stats.penalty_2m_after > page_tie_stats.penalty_2m_before
        || (page_tie_stats.penalty_2m_after == page_tie_stats.penalty_2m_before
            && page_tie_stats.penalty_4k_after > page_tie_stats.penalty_4k_before)) return 34;
    sparse_domain_page_pool.run(
        low_jobs, main_auth, block_auth, storage, layout, sparse, mod);
    if (!compare_factor("sparse-domain-page-1", main_auth, block_auth,
                        main_states, block_states, low_rm, low_rd, storage, layout)) return 35;
    sparse_domain_page_pool.run(
        low_jobs, main_auth, block_auth, storage, layout, sparse, mod);
    if (!compare_factor("sparse-domain-page-2", main_auth, block_auth,
                        main_states, block_states, low2_rm, low2_rd, storage, layout)) return 36;

    fill_factor(main_auth, block_auth, main_states, block_states, init_m, init_d, storage, layout);
    CpuLowSparsePersistentPool sparse_domain_unrefined_pool(
        2, CPU_LOW_SCHEDULE_DOMAIN, 1, false);
    sparse_domain_unrefined_pool.run(
        low_jobs, main_auth, block_auth, storage, layout, sparse, mod);
    if (!compare_factor("sparse-domain-unrefined-1", main_auth, block_auth,
                        main_states, block_states, low_rm, low_rd, storage, layout)) return 31;
    if (sparse_domain_unrefined_pool.domain_refined_boundaries != 0
        || sparse_domain_unrefined_pool.domain_refined_job_moves != 0) return 32;
    sparse_domain_unrefined_pool.run(
        low_jobs, main_auth, block_auth, storage, layout, sparse, mod);
    if (!compare_factor("sparse-domain-unrefined-2", main_auth, block_auth,
                        main_states, block_states, low2_rm, low2_rd, storage, layout)) return 33;

    size_t sparse_orbit_ops = sparse.orbit_count();
    if (!sparse_orbit_ops
        || sparse_orbit_ops != sparse.nn_orbit_ops.size() + sparse.nr_orbit_ops.size() + sparse.nl_orbit_ops.size()) {
        std::cerr << "FAIL sparse split orbit accounting\n";
        return 14;
    }

    auto [high_rm, high_rd] = reference_window(
        W, W - 1, LOW_LUT_K + 1, mod,
        main_states, block_states, mi, di, init_m, init_d);
    auto [high2_rm, high2_rd] = reference_window(
        W, W - 1, LOW_LUT_K + 1, mod,
        main_states, block_states, mi, di, high_rm, high_rd);
    WindowPlan high_wp = make_direct2d_window(true);
    auto high_jobs = make_cpu_high_jobs(W, high_wp);
    std::vector<const CpuHighJob*> high_job_ptrs;
    high_job_ptrs.reserve(high_jobs.size());
    for (const auto& job : high_jobs)
        if (job.main_size || job.block_size) high_job_ptrs.push_back(&job);

    fill_factor(main_auth, block_auth, main_states, block_states, init_m, init_d, storage, layout);
    CpuHighPool high_pool(2);
    high_pool.run(high_job_ptrs, main_auth, block_auth, storage, layout, highdesc, highcross, mod);
    if (!compare_factor("cpu-high", main_auth, block_auth, main_states, block_states,
                        high_rm, high_rd, storage, layout)) return 15;

    fill_factor(main_auth, block_auth, main_states, block_states, init_m, init_d, storage, layout);
    CpuHighDirectPool high_direct_pool(2);
    high_direct_pool.run(high_job_ptrs, main_auth, block_auth,
                         storage, layout, highdirect, highcross, mod);
    if (!compare_factor("cpu-high-direct", main_auth, block_auth, main_states, block_states,
                        high_rm, high_rd, storage, layout)) return 16;

    fill_factor(main_auth, block_auth, main_states, block_states, init_m, init_d, storage, layout);
    CpuHighDirectPersistentPool high_persistent_pool(2);
    high_persistent_pool.run(high_job_ptrs, main_auth, block_auth,
                             storage, layout, highdirect, highcross, mod);
    if (!compare_factor("cpu-high-persistent-1", main_auth, block_auth,
                        main_states, block_states, high_rm, high_rd,
                        storage, layout)) return 17;
    high_persistent_pool.run(high_job_ptrs, main_auth, block_auth,
                             storage, layout, highdirect, highcross, mod);
    if (!compare_factor("cpu-high-persistent-2", main_auth, block_auth,
                        main_states, block_states, high2_rm, high2_rd,
                        storage, layout)) return 18;

    std::vector<const CpuHighJob*> high_jobs_a, high_jobs_b;
    for (const CpuHighJob* job : high_job_ptrs) {
        ((job->mask & 1u) ? high_jobs_a : high_jobs_b).push_back(job);
    }
    fill_factor(main_auth, block_auth, main_states, block_states, init_m, init_d, storage, layout);
    CpuHighDirectPool high_parallel_a(1), high_parallel_b(1);
    std::thread ta([&] {
        high_parallel_a.run(high_jobs_a, main_auth, block_auth,
                            storage, layout, highdirect, highcross, mod);
    });
    std::thread tb([&] {
        high_parallel_b.run(high_jobs_b, main_auth, block_auth,
                            storage, layout, highdirect, highcross, mod);
    });
    ta.join(); tb.join();
    if (!compare_factor("cpu-high-parallel-partition", main_auth, block_auth,
                        main_states, block_states, high_rm, high_rd,
                        storage, layout)) return 19;

    double sparse_meta_mib = double(
        sparse_orbit_ops*sizeof(CpuLowSparseOrbitOp)
        + sparse.local_closure_ops.size()*sizeof(CpuLowSparseClosureOp)
        + sparse.cross_closure_ops.size()*sizeof(CpuLowSparseClosureOp)
        + sparse.high_cross_rank.size()*sizeof(uint16_t))/(1<<20);

    std::cout << "cpu-low-selftest OK W=" << W
              << " main=" << main_states.size() << " block=" << block_states.size()
              << " refine_before=" << refine_before
              << " refine_after=" << refine_after
              << " refine_boundaries=" << refine_boundaries
              << " refine_moves=" << refine_moves
              << " out_groups=" << out_pool.groups() << " in_groups=" << in_pool.groups()
              << " direct_groups=" << direct_pool.groups() << " sparse_groups=" << sparse_pool.groups()
              << " sparse_persistent_groups=" << sparse_persistent_pool.groups()
              << " sparse_persistent_start_s=" << sparse_persistent_pool.worker_start_s
              << " sparse_sticky_groups=" << sparse_sticky_pool.groups()
              << " sparse_sticky_build_s=" << sparse_sticky_pool.schedule_build_s
              << " sparse_contiguous_groups=" << sparse_contiguous_pool.groups()
              << " sparse_contiguous_build_s=" << sparse_contiguous_pool.schedule_build_s
              << " sparse_contiguous_optimal_cap=" << sparse_contiguous_pool.contiguous_optimal_cap
              << " sparse_domain_groups=" << sparse_domain_pool.groups()
              << " sparse_domain_build_s=" << sparse_domain_pool.schedule_build_s
              << " sparse_domain_size=" << sparse_domain_pool.domain_size
              << " sparse_domain_refine=" << int(sparse_domain_pool.domain_refine)
              << " sparse_domain_normalized_cap=" << sparse_domain_pool.domain_normalized_cap
              << " sparse_domain_active_domains=" << sparse_domain_pool.domain_active_domains
              << " sparse_domain_refined_boundaries=" << sparse_domain_pool.domain_refined_boundaries
              << " sparse_domain_refined_job_moves=" << sparse_domain_pool.domain_refined_job_moves
              << " sparse_domain_page_groups=" << sparse_domain_page_pool.groups()
              << " sparse_domain_page_boundary_moves=" << page_tie_stats.boundary_moves
              << " sparse_domain_page_moved_jobs=" << page_tie_stats.moved_jobs
              << " sparse_domain_page_penalty_2m_before=" << page_tie_stats.penalty_2m_before
              << " sparse_domain_page_penalty_2m_after=" << page_tie_stats.penalty_2m_after
              << " sparse_domain_page_penalty_4k_before=" << page_tie_stats.penalty_4k_before
              << " sparse_domain_page_penalty_4k_after=" << page_tie_stats.penalty_4k_after
              << " sparse_domain_page_max_before=" << page_tie_stats.max_worker_cells_before
              << " sparse_domain_page_max_after=" << page_tie_stats.max_worker_cells_after
              << " sparse_domain_unrefined_groups=" << sparse_domain_unrefined_pool.groups()
              << " sparse_domain_unrefined_refine=" << int(sparse_domain_unrefined_pool.domain_refine)
              << " sparse_domain_unrefined_boundaries=" << sparse_domain_unrefined_pool.domain_refined_boundaries
              << " sparse_domain_unrefined_moves=" << sparse_domain_unrefined_pool.domain_refined_job_moves
              << " cpu_high_groups=" << high_pool.groups()
              << " cpu_high_direct_groups=" << high_direct_pool.groups()
              << " cpu_high_persistent_groups=" << high_persistent_pool.groups()
              << " cpu_high_persistent_start_s=" << high_persistent_pool.worker_start_s
              << " cpu_high_parallel_groups="
              << (high_parallel_a.groups()+high_parallel_b.groups())
              << " out_scratch_mib=" << double(out_pool.peak_scratch_bytes()) / (1 << 20)
              << " in_scratch_mib=" << double(in_pool.peak_scratch_bytes()) / (1 << 20)
              << " cpu_high_scratch_mib=" << double(high_pool.peak_scratch_bytes()) / (1 << 20)
              << " direct_scratch_mib=0 sparse_scratch_mib=0 cpu_high_direct_scratch_mib=0"
              << " dense_orbit_mib=" << double(orbit.rec.size() * sizeof(uint64_t)) / (1 << 20)
              << " sparse_nn_orbit_ops=" << sparse.nn_orbit_ops.size()
              << " sparse_nr_orbit_ops=" << sparse.nr_orbit_ops.size()
              << " sparse_nl_orbit_ops=" << sparse.nl_orbit_ops.size()
              << " sparse_local_closure_ops=" << sparse.local_closure_ops.size()
              << " sparse_cross_closure_ops=" << sparse.cross_closure_ops.size()
              << " sparse_meta_mib=" << sparse_meta_mib
              << " cpu_high_cross_mib="
              << double(highcross.low_cross_rank.size()*sizeof(uint16_t))/(1<<20)
              << " cpu_high_direct_orbit_mib="
              << double(highdirect.orbit_ops.size()*sizeof(CpuHighOrbitOp))/(1<<20)
              << " cpu_high_direct_closure_mib="
              << double(highdirect.closure_ops.size()*sizeof(CpuHighClosureOp))/(1<<20)
              << '\n';

    sparse_persistent_pool.shutdown();
    sparse_sticky_pool.shutdown();
    sparse_contiguous_pool.shutdown();
    sparse_domain_pool.shutdown();
    sparse_domain_page_pool.shutdown();
    sparse_domain_unrefined_pool.shutdown();
    high_persistent_pool.shutdown();
    high_pool.release();
    in_pool.release(); out_pool.release();
    main_auth.release(); block_auth.release();
    return 0;
}
