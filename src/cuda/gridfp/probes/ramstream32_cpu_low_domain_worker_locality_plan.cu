#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <unordered_set>
#include <vector>

#define RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../oneesan_cuda_gridfp_ramstream32_factorized_bidesc_compact.cu"
#undef RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../ramstream32_cpu_low_domain_page.hpp"
#include "../ramstream32_cpu_low_domain_worker_locality.hpp"
#include "../ramstream32_cpu_low_domain_worker_coalesce.hpp"
#include "../ramstream32_cpu_low_domain_worker_unique_coalesce.hpp"

struct OwnerPageMetrics {
    uint64_t cross_4k = 0;
    uint64_t cross_2m = 0;
    uint64_t owner_transitions = 0;
};

static void add_owner_boundary(
    std::unordered_set<uint64_t>& p4,
    std::unordered_set<uint64_t>& p2,
    uint64_t boundary_byte,
    uint64_t array_tag
) {
    constexpr uint64_t PAGE4K = 4096ull;
    constexpr uint64_t PAGE2M = 2ull << 20;
    constexpr uint64_t TAG = 1ull << 63;
    uint64_t tag = array_tag ? TAG : 0;
    if (boundary_byte % PAGE4K) p4.insert(tag | (boundary_byte / PAGE4K));
    if (boundary_byte % PAGE2M) p2.insert(tag | (boundary_byte / PAGE2M));
}

static void scan_owner_pages(
    const std::vector<StorageBlock>& blocks,
    const StorageFactorHost& storage,
    const std::vector<int>& owner,
    uint64_t array_tag,
    std::unordered_set<uint64_t>& p4,
    std::unordered_set<uint64_t>& p2,
    uint64_t& transitions
) {
    constexpr int S = FactorTablesHost::STRIDE;
    const uint32_t nmasks = uint32_t(1) << HIGH_LUT_K;
    for (const StorageBlock& sb : blocks) {
        if (!sb.valid || !sb.rows || !sb.cols) continue;
        bool have_prev = false;
        uint64_t prev_end = 0;
        int prev_owner = -1;
        for (uint32_t mask = 0; mask < nmasks; ++mask) {
            size_t ix = size_t(mask) * S + sb.he;
            uint32_t rows = G_FACTOR.high_mask_off[ix + 1] - G_FACTOR.high_mask_off[ix];
            if (!rows) continue;
            if (owner[mask] < 0) {
                std::cerr << "worker-locality page plan missing owner mask=" << mask << '\n';
                std::exit(176);
            }
            uint32_t row0 = storage.high_mask_begin[
                size_t(mask) * StorageFactorHost::S + sb.he];
            uint64_t begin_elem = uint64_t(sb.off) + uint64_t(row0) * sb.cols;
            uint64_t begin_byte = begin_elem * sizeof(Count);
            uint64_t end_byte = begin_byte + uint64_t(rows) * sb.cols * sizeof(Count);
            if (have_prev) {
                if (begin_byte != prev_end) {
                    std::cerr << "worker-locality storage mask ranges not contiguous\n";
                    std::exit(177);
                }
                if (owner[mask] != prev_owner) {
                    ++transitions;
                    add_owner_boundary(p4, p2, begin_byte, array_tag);
                }
            }
            have_prev = true;
            prev_end = end_byte;
            prev_owner = owner[mask];
        }
    }
}

static OwnerPageMetrics owner_page_metrics(
    const StorageLayout& layout,
    const StorageFactorHost& storage,
    const std::vector<int>& owner
) {
    std::unordered_set<uint64_t> p4, p2;
    uint64_t transitions = 0;
    scan_owner_pages(
        layout.main_blocks, storage, owner, 0, p4, p2, transitions);
    scan_owner_pages(
        layout.block_blocks, storage, owner, 1, p4, p2, transitions);
    return {uint64_t(p4.size()), uint64_t(p2.size()), transitions};
}

static std::vector<int> worker_owner(
    const CpuLowSparsePersistentPool& pool,
    const std::vector<CpuLowJob>& jobs
) {
    std::vector<int> owner(size_t(1) << HIGH_LUT_K, -1);
    for (int w = 0; w < pool.workers; ++w) {
        for (size_t q : pool.sticky_worker_jobs[size_t(w)]) {
            uint32_t mask = jobs[q].mask;
            if (owner[mask] >= 0) {
                std::cerr << "worker-locality duplicate mask owner\n";
                std::exit(178);
            }
            owner[mask] = w;
        }
    }
    return owner;
}

static std::vector<int> domain_owner(
    const std::vector<int>& worker, int domain_size
) {
    std::vector<int> out(worker.size(), -1);
    for (size_t i = 0; i < worker.size(); ++i)
        if (worker[i] >= 0) out[i] = worker[i] / domain_size;
    return out;
}

static uint64_t max_cells(const CpuLowSparsePersistentPool& pool) {
    return pool.sticky_worker_cells.empty() ? 0
        : *std::max_element(
            pool.sticky_worker_cells.begin(), pool.sticky_worker_cells.end());
}

static uint64_t total_cells(const CpuLowSparsePersistentPool& pool) {
    return std::accumulate(
        pool.sticky_worker_cells.begin(), pool.sticky_worker_cells.end(), uint64_t(0));
}

static bool same_metrics(const OwnerPageMetrics& a, const OwnerPageMetrics& b) {
    return a.cross_2m == b.cross_2m
        && a.cross_4k == b.cross_4k
        && a.owner_transitions == b.owner_transitions;
}

int main(int argc, char** argv) {
    int n = argc > 1 ? std::atoi(argv[1]) : TARGET_W - 1;
    int workers = argc > 2 ? std::max(1, std::atoi(argv[2])) : 64;
    int domain_size = argc > 3 ? std::atoi(argv[3]) : 32;
    if (n < 2 || n + 1 != TARGET_W || n + 1 > MAXW) return 1;
    if (workers <= 0 || domain_size <= 0 || domain_size > workers) return 1;
    if constexpr (LOW_LUT_K + HIGH_LUT_K != TARGET_W - 1) return 1;

    build_full_dp();
    G_FACTOR = build_factor_tables();
    StorageFactorHost storage = build_storage_factor_tables(G_FACTOR);
    StorageLayout layout = build_storage_layout(storage);
    LowDescHost lowdesc = build_low_descriptors(storage, layout);
    LowOrbitHost orbit = build_cpu_low_orbit(storage, layout, lowdesc);
    CpuLowSparseHost sparse = build_cpu_low_sparse(storage, layout, lowdesc, orbit);
    WindowPlan low_wp = make_direct2d_window(false);
    auto jobs = make_cpu_low_jobs(n + 1, low_wp);

    CpuLowSparsePersistentPool baseline(
        workers, CPU_LOW_SCHEDULE_DOMAIN, domain_size, true);
    CpuLowSparsePersistentPool locality(
        workers, CPU_LOW_SCHEDULE_DOMAIN, domain_size, true);
    CpuLowSparsePersistentPool coalesced(
        workers, CPU_LOW_SCHEDULE_DOMAIN, domain_size, true);
    CpuLowSparsePersistentPool unique(
        workers, CPU_LOW_SCHEDULE_DOMAIN, domain_size, true);
    baseline.prepare_static_schedule(jobs, sparse);
    locality.prepare_static_schedule(jobs, sparse);
    coalesced.prepare_static_schedule(jobs, sparse);
    unique.prepare_static_schedule(jobs, sparse);

    CpuLowDomainPageTieStats baseline_page = cpu_low_apply_domain_page_tiebreak(
        baseline, jobs, sparse, storage, layout);
    CpuLowDomainPageTieStats locality_page = cpu_low_apply_domain_page_tiebreak(
        locality, jobs, sparse, storage, layout);
    CpuLowDomainPageTieStats coalesced_page = cpu_low_apply_domain_page_tiebreak(
        coalesced, jobs, sparse, storage, layout);
    CpuLowDomainPageTieStats unique_page = cpu_low_apply_domain_page_tiebreak(
        unique, jobs, sparse, storage, layout);

    CpuLowDomainWorkerLocalityStats loc =
        cpu_low_apply_domain_worker_locality(locality, jobs, sparse);
    CpuLowDomainWorkerLocalityStats coal_parent =
        cpu_low_apply_domain_worker_locality(coalesced, jobs, sparse);
    CpuLowDomainWorkerLocalityStats unique_parent =
        cpu_low_apply_domain_worker_locality(unique, jobs, sparse);
    CpuLowDomainWorkerCoalesceStats coal =
        cpu_low_apply_domain_worker_coalesce(
            coalesced, jobs, sparse, storage, layout);
    CpuLowDomainWorkerUniqueCoalesceStats uniq =
        cpu_low_apply_domain_worker_unique_coalesce(
            unique, jobs, sparse, storage, layout);

    uint64_t baseline_total = total_cells(baseline);
    uint64_t locality_total = total_cells(locality);
    uint64_t coalesced_total = total_cells(coalesced);
    uint64_t unique_total = total_cells(unique);
    if (baseline_total != locality_total || baseline_total != coalesced_total
        || baseline_total != unique_total) {
        std::cerr << "worker-coalesce total-cell mismatch\n";
        return 2;
    }

    uint64_t baseline_max = max_cells(baseline);
    uint64_t locality_max = max_cells(locality);
    uint64_t coalesced_max = max_cells(coalesced);
    uint64_t unique_max = max_cells(unique);
    if (locality_max > baseline_max || coalesced_max > locality_max
        || unique_max > locality_max) {
        std::cerr << "worker-coalesce max-worker regression"
                  << " baseline=" << baseline_max
                  << " locality=" << locality_max
                  << " coalesced=" << coalesced_max
                  << " unique=" << unique_max << '\n';
        return 3;
    }

    auto base_worker_owner = worker_owner(baseline, jobs);
    auto loc_worker_owner = worker_owner(locality, jobs);
    auto coal_worker_owner = worker_owner(coalesced, jobs);
    auto unique_worker_owner = worker_owner(unique, jobs);
    OwnerPageMetrics base_worker_pages = owner_page_metrics(
        layout, storage, base_worker_owner);
    OwnerPageMetrics loc_worker_pages = owner_page_metrics(
        layout, storage, loc_worker_owner);
    OwnerPageMetrics coal_worker_pages = owner_page_metrics(
        layout, storage, coal_worker_owner);
    OwnerPageMetrics unique_worker_pages = owner_page_metrics(
        layout, storage, unique_worker_owner);
    OwnerPageMetrics base_domain_pages = owner_page_metrics(
        layout, storage, domain_owner(base_worker_owner, domain_size));
    OwnerPageMetrics loc_domain_pages = owner_page_metrics(
        layout, storage, domain_owner(loc_worker_owner, domain_size));
    OwnerPageMetrics coal_domain_pages = owner_page_metrics(
        layout, storage, domain_owner(coal_worker_owner, domain_size));
    OwnerPageMetrics unique_domain_pages = owner_page_metrics(
        layout, storage, domain_owner(unique_worker_owner, domain_size));

    if (!same_metrics(base_domain_pages, loc_domain_pages)
        || !same_metrics(base_domain_pages, coal_domain_pages)
        || !same_metrics(base_domain_pages, unique_domain_pages)) {
        std::cerr << "worker-coalesce changed domain boundaries\n";
        return 4;
    }
    if (coal.penalty_2m_after > coal.penalty_2m_before
        || (coal.penalty_2m_after == coal.penalty_2m_before
            && coal.penalty_4k_after > coal.penalty_4k_before)
        || (coal.penalty_2m_after == coal.penalty_2m_before
            && coal.penalty_4k_after == coal.penalty_4k_before
            && coal.owner_transitions_after > coal.owner_transitions_before)) {
        std::cerr << "worker-coalesce internal objective regression\n";
        return 5;
    }

    CpuLowDomainWorkerUniqueScore uniq_before{
        uniq.unique_pages_2m_before,
        uniq.unique_pages_4k_before,
        uniq.owner_transitions_before};
    CpuLowDomainWorkerUniqueScore uniq_after{
        uniq.unique_pages_2m_after,
        uniq.unique_pages_4k_after,
        uniq.owner_transitions_after};
    if (cpu_low_worker_unique_score_less(uniq_before, uniq_after)) {
        std::cerr << "worker-unique exact objective regression\n";
        return 6;
    }
    if (uniq.unique_pages_2m_before != loc_worker_pages.cross_2m
        || uniq.unique_pages_4k_before != loc_worker_pages.cross_4k
        || uniq.owner_transitions_before != loc_worker_pages.owner_transitions
        || uniq.unique_pages_2m_after != unique_worker_pages.cross_2m
        || uniq.unique_pages_4k_after != unique_worker_pages.cross_4k
        || uniq.owner_transitions_after != unique_worker_pages.owner_transitions) {
        std::cerr << "worker-unique internal/external page metric mismatch\n";
        return 7;
    }

    double avg = workers ? double(baseline_total) / workers : 0.0;
    std::cout << std::setprecision(12)
              << "cpu_low_domain_worker_locality_plan OK"
              << " objective=global-unique-neighbor-coalesce-v5.27-plan"
              << " local_objective=neighbor-page-coalesce-under-domain-cap-v5.26-plan"
              << " parent_objective=contiguous-under-lpt-cap-v5.25-plan"
              << " n=" << n
              << " workers=" << workers
              << " domain_size=" << domain_size
              << " domains=" << loc.domains
              << " total_cells=" << baseline_total
              << " baseline_max_worker_cells=" << baseline_max
              << " locality_max_worker_cells=" << locality_max
              << " coalesced_max_worker_cells=" << coalesced_max
              << " unique_max_worker_cells=" << unique_max
              << " baseline_imbalance=" << (avg ? double(baseline_max) / avg : 0.0)
              << " locality_imbalance=" << (avg ? double(locality_max) / avg : 0.0)
              << " coalesced_imbalance=" << (avg ? double(coalesced_max) / avg : 0.0)
              << " unique_imbalance=" << (avg ? double(unique_max) / avg : 0.0)
              << " converted_domains=" << loc.converted_domains
              << " fallback_domains=" << loc.fallback_domains
              << " unchanged_trivial_domains=" << loc.unchanged_trivial_domains
              << " converted_jobs=" << loc.converted_jobs
              << " contiguous_worker_segments=" << loc.contiguous_worker_segments
              << " coalesce_noncontiguous_domains_before="
              << coal.noncontiguous_domains_before
              << " coalesce_improved_domains=" << coal.improved_domains
              << " coalesce_accepted_moves=" << coal.accepted_moves
              << " coalesce_cap_rejections=" << coal.cap_rejections
              << " unique_noncontiguous_domains_before="
              << uniq.noncontiguous_domains_before
              << " unique_improved_domains=" << uniq.improved_domains
              << " unique_accepted_moves=" << uniq.accepted_moves
              << " unique_cap_rejections=" << uniq.cap_rejections
              << " unique_page_improving_moves=" << uniq.unique_page_improving_moves
              << " unique_transition_only_moves=" << uniq.transition_only_moves
              << " unique_internal_pages_2m_before=" << uniq.unique_pages_2m_before
              << " unique_internal_pages_2m_after=" << uniq.unique_pages_2m_after
              << " unique_internal_pages_4k_before=" << uniq.unique_pages_4k_before
              << " unique_internal_pages_4k_after=" << uniq.unique_pages_4k_after
              << " baseline_cross_worker_pages_2m=" << base_worker_pages.cross_2m
              << " locality_cross_worker_pages_2m=" << loc_worker_pages.cross_2m
              << " coalesced_cross_worker_pages_2m=" << coal_worker_pages.cross_2m
              << " unique_cross_worker_pages_2m=" << unique_worker_pages.cross_2m
              << " baseline_cross_worker_pages_4k=" << base_worker_pages.cross_4k
              << " locality_cross_worker_pages_4k=" << loc_worker_pages.cross_4k
              << " coalesced_cross_worker_pages_4k=" << coal_worker_pages.cross_4k
              << " unique_cross_worker_pages_4k=" << unique_worker_pages.cross_4k
              << " baseline_worker_owner_transitions=" << base_worker_pages.owner_transitions
              << " locality_worker_owner_transitions=" << loc_worker_pages.owner_transitions
              << " coalesced_worker_owner_transitions=" << coal_worker_pages.owner_transitions
              << " unique_worker_owner_transitions=" << unique_worker_pages.owner_transitions
              << " cross_domain_pages_2m=" << base_domain_pages.cross_2m
              << " cross_domain_pages_4k=" << base_domain_pages.cross_4k
              << " cross_domain_owner_transitions=" << base_domain_pages.owner_transitions
              << " baseline_page_moves=" << baseline_page.boundary_moves
              << " locality_page_moves=" << locality_page.boundary_moves
              << " coalesced_page_moves=" << coalesced_page.boundary_moves
              << " unique_page_moves=" << unique_page.boundary_moves
              << " worker_locality_build_s=" << loc.build_s
              << " coalesced_parent_build_s=" << coal_parent.build_s
              << " unique_parent_build_s=" << unique_parent.build_s
              << " worker_coalesce_build_s=" << coal.build_s
              << " worker_unique_coalesce_build_s=" << uniq.build_s
              << '\n';
    return 0;
}
