#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <unordered_set>
#include <vector>

#define RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../oneesan_cuda_gridfp_ramstream32_factorized_bidesc_compact.cu"
#undef RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../ramstream32_cpu_low_sparse_persistent.hpp"

struct CpuLowBoundaryPages {
    std::unordered_set<uint64_t> shared_4k;
    std::unordered_set<uint64_t> cross_worker_4k;
    std::unordered_set<uint64_t> shared_2m;
    std::unordered_set<uint64_t> cross_worker_2m;
};

struct CpuLowPageMetrics {
    uint64_t shared_4k = 0;
    uint64_t cross_4k = 0;
    uint64_t shared_2m = 0;
    uint64_t cross_2m = 0;
    double cross_of_shared_4k = 0.0;
    double cross_of_shared_2m = 0.0;
    double cross_auth_4k = 0.0;
    double cross_auth_2m = 0.0;
};

static void cpu_low_add_boundary_page(
    CpuLowBoundaryPages& out, uint64_t boundary_byte,
    int left_owner, int right_owner
) {
    constexpr uint64_t PAGE4K = 4096ull;
    constexpr uint64_t PAGE2M = 2ull << 20;
    if (boundary_byte % PAGE4K) {
        uint64_t page = boundary_byte / PAGE4K;
        out.shared_4k.insert(page);
        if (left_owner != right_owner) out.cross_worker_4k.insert(page);
    }
    if (boundary_byte % PAGE2M) {
        uint64_t page = boundary_byte / PAGE2M;
        out.shared_2m.insert(page);
        if (left_owner != right_owner) out.cross_worker_2m.insert(page);
    }
}

static CpuLowBoundaryPages cpu_low_boundary_pages(
    const std::vector<StorageBlock>& blocks,
    const StorageFactorHost& storage,
    const std::vector<int>& owner
) {
    constexpr int S = FactorTablesHost::STRIDE;
    const uint32_t nmasks = 1u << HIGH_LUT_K;
    CpuLowBoundaryPages out;

    for (const StorageBlock& sb : blocks) {
        if (!sb.valid || !sb.rows || !sb.cols) continue;
        bool have_prev = false;
        uint64_t prev_end_byte = 0;
        int prev_owner = -1;

        for (uint32_t mask = 0; mask < nmasks; ++mask) {
            size_t ix = size_t(mask) * S + sb.he;
            uint32_t rows = G_FACTOR.high_mask_off[ix + 1] - G_FACTOR.high_mask_off[ix];
            if (!rows) continue;
            if (owner[mask] < 0) {
                std::cerr << "cpu LOW page plan missing owner mask=" << mask
                          << " h=" << unsigned(sb.he) << '\n';
                std::exit(3);
            }

            uint32_t row0 = storage.high_mask_begin[
                size_t(mask) * StorageFactorHost::S + sb.he];
            uint64_t begin_elem = uint64_t(sb.off) + uint64_t(row0) * sb.cols;
            uint64_t begin_byte = begin_elem * sizeof(Count);
            uint64_t end_byte = begin_byte + uint64_t(rows) * sb.cols * sizeof(Count);

            if (have_prev) {
                if (begin_byte != prev_end_byte) {
                    std::cerr << "cpu LOW storage mask ranges not contiguous h="
                              << unsigned(sb.he) << " begin=" << begin_byte
                              << " previous_end=" << prev_end_byte << '\n';
                    std::exit(4);
                }
                cpu_low_add_boundary_page(out, begin_byte, prev_owner, owner[mask]);
            }
            have_prev = true;
            prev_end_byte = end_byte;
            prev_owner = owner[mask];
        }
    }
    return out;
}

static uint64_t cpu_low_page_count(uint64_t bytes, uint64_t page) {
    return bytes ? (bytes + page - 1) / page : 0;
}

static CpuLowPageMetrics cpu_low_page_metrics(
    const StorageLayout& layout, const StorageFactorHost& storage,
    const std::vector<int>& owner
) {
    CpuLowBoundaryPages main_pages = cpu_low_boundary_pages(
        layout.main_blocks, storage, owner);
    CpuLowBoundaryPages block_pages = cpu_low_boundary_pages(
        layout.block_blocks, storage, owner);

    CpuLowPageMetrics m;
    m.shared_4k = main_pages.shared_4k.size() + block_pages.shared_4k.size();
    m.cross_4k = main_pages.cross_worker_4k.size() + block_pages.cross_worker_4k.size();
    m.shared_2m = main_pages.shared_2m.size() + block_pages.shared_2m.size();
    m.cross_2m = main_pages.cross_worker_2m.size() + block_pages.cross_worker_2m.size();

    uint64_t main_bytes = uint64_t(layout.main_size) * sizeof(Count);
    uint64_t block_bytes = uint64_t(layout.block_size) * sizeof(Count);
    uint64_t total_pages_4k = cpu_low_page_count(main_bytes, 4096)
        + cpu_low_page_count(block_bytes, 4096);
    uint64_t total_pages_2m = cpu_low_page_count(main_bytes, 2ull << 20)
        + cpu_low_page_count(block_bytes, 2ull << 20);

    m.cross_of_shared_4k = m.shared_4k ? double(m.cross_4k) / m.shared_4k : 0.0;
    m.cross_of_shared_2m = m.shared_2m ? double(m.cross_2m) / m.shared_2m : 0.0;
    m.cross_auth_4k = total_pages_4k ? double(m.cross_4k) / total_pages_4k : 0.0;
    m.cross_auth_2m = total_pages_2m ? double(m.cross_2m) / total_pages_2m : 0.0;
    return m;
}

static std::vector<int> cpu_low_owner_from_pool(
    const CpuLowSparsePersistentPool& pool,
    const std::vector<CpuLowJob>& jobs
) {
    std::vector<int> owner(size_t(1) << HIGH_LUT_K, -1);
    for (int w = 0; w < pool.workers; ++w) {
        for (size_t q : pool.sticky_worker_jobs[size_t(w)]) {
            uint32_t mask = jobs[q].mask;
            if (owner[mask] >= 0) {
                std::cerr << "cpu LOW duplicate mask owner mask=" << mask
                          << " old=" << owner[mask] << " new=" << w << '\n';
                std::exit(5);
            }
            owner[mask] = w;
        }
    }
    return owner;
}

static std::vector<int> cpu_low_domain_owner(
    const std::vector<int>& worker_owner, int domain_size
) {
    std::vector<int> out(worker_owner.size(), -1);
    if (domain_size <= 0) return out;
    for (size_t i = 0; i < worker_owner.size(); ++i) {
        if (worker_owner[i] >= 0) out[i] = worker_owner[i] / domain_size;
    }
    return out;
}

static uint64_t cpu_low_min_cells(const std::vector<uint64_t>& cells) {
    if (cells.empty()) return 0;
    return *std::min_element(cells.begin(), cells.end());
}

static uint64_t cpu_low_max_cells(const std::vector<uint64_t>& cells) {
    return cells.empty() ? 0 : *std::max_element(cells.begin(), cells.end());
}

static size_t cpu_low_assigned_jobs(
    const std::vector<std::vector<size_t>>& worker_jobs
) {
    size_t z = 0;
    for (const auto& x : worker_jobs) z += x.size();
    return z;
}

int main(int argc, char** argv) {
    int n = argc > 1 ? std::atoi(argv[1]) : TARGET_W - 1;
    int workers = argc > 2 ? std::max(1, std::atoi(argv[2])) : 32;
    bool dump_workers = false;
    int domain_size = 0;
    for (int i = 3; i < argc; ++i) {
        if (std::strcmp(argv[i], "--workers") == 0) {
            dump_workers = true;
        } else if (std::strcmp(argv[i], "--domain-size") == 0) {
            if (++i >= argc) {
                std::cerr << "--domain-size requires a positive integer\n";
                return 1;
            }
            domain_size = std::atoi(argv[i]);
            if (domain_size <= 0 || domain_size > workers) {
                std::cerr << "--domain-size must be in 1..workers\n";
                return 1;
            }
        } else {
            std::cerr << "unknown argument: " << argv[i] << '\n';
            return 1;
        }
    }

    int W = n + 1;
    if (W != TARGET_W || n < 2 || W > MAXW || workers <= 0) return 1;
    if constexpr (LOW_LUT_K + HIGH_LUT_K != TARGET_W - 1) return 1;

    build_full_dp();
    G_FACTOR = build_factor_tables();
    StorageFactorHost storage = build_storage_factor_tables(G_FACTOR);
    StorageLayout layout = build_storage_layout(storage);
    LowDescHost lowdesc = build_low_descriptors(storage, layout);
    LowOrbitHost orbit = build_cpu_low_orbit(storage, layout, lowdesc);
    CpuLowSparseHost sparse = build_cpu_low_sparse(storage, layout, lowdesc, orbit);

    WindowPlan low_wp = make_direct2d_window(false);
    auto jobs = make_cpu_low_jobs(W, low_wp);

    CpuLowSparsePersistentPool lpt_pool(workers, CPU_LOW_SCHEDULE_STICKY);
    CpuLowSparsePersistentPool contiguous_pool(workers, CPU_LOW_SCHEDULE_CONTIGUOUS);
    CpuLowSparsePersistentPool domain_pool(
        workers, CPU_LOW_SCHEDULE_DOMAIN, domain_size > 0 ? domain_size : workers);
    lpt_pool.prepare_static_schedule(jobs, sparse);
    contiguous_pool.prepare_static_schedule(jobs, sparse);
    if (domain_size > 0) domain_pool.prepare_static_schedule(jobs, sparse);

    size_t nonempty_jobs = 0;
    uint64_t exact_total = 0;
    for (const auto& job : jobs) {
        if (!job.main_size && !job.block_size) continue;
        ++nonempty_jobs;
        exact_total += cpu_low_sparse_job_cells(job, sparse);
    }

    uint64_t lpt_total = std::accumulate(
        lpt_pool.sticky_worker_cells.begin(), lpt_pool.sticky_worker_cells.end(), uint64_t(0));
    uint64_t contiguous_total = std::accumulate(
        contiguous_pool.sticky_worker_cells.begin(),
        contiguous_pool.sticky_worker_cells.end(), uint64_t(0));
    size_t lpt_jobs = cpu_low_assigned_jobs(lpt_pool.sticky_worker_jobs);
    size_t contiguous_jobs = cpu_low_assigned_jobs(contiguous_pool.sticky_worker_jobs);
    if (lpt_total != exact_total || contiguous_total != exact_total
        || lpt_jobs != nonempty_jobs || contiguous_jobs != nonempty_jobs) {
        std::cerr << "cpu LOW static plan accounting mismatch"
                  << " expected_jobs=" << nonempty_jobs
                  << " lpt_jobs=" << lpt_jobs
                  << " contiguous_jobs=" << contiguous_jobs
                  << " expected_cells=" << exact_total
                  << " lpt_cells=" << lpt_total
                  << " contiguous_cells=" << contiguous_total << '\n';
        return 2;
    }

    uint64_t min_cells = cpu_low_min_cells(lpt_pool.sticky_worker_cells);
    uint64_t max_cells = cpu_low_max_cells(lpt_pool.sticky_worker_cells);
    uint64_t contiguous_min_cells = cpu_low_min_cells(contiguous_pool.sticky_worker_cells);
    uint64_t contiguous_max_cells = cpu_low_max_cells(contiguous_pool.sticky_worker_cells);
    double avg = workers ? double(exact_total) / workers : 0.0;
    double imbalance = avg > 0.0 ? double(max_cells) / avg : 0.0;
    double contiguous_imbalance = avg > 0.0
        ? double(contiguous_max_cells) / avg : 0.0;

    std::vector<int> lpt_owner = cpu_low_owner_from_pool(lpt_pool, jobs);
    std::vector<int> contiguous_owner = cpu_low_owner_from_pool(contiguous_pool, jobs);
    CpuLowPageMetrics lpt_pages = cpu_low_page_metrics(layout, storage, lpt_owner);
    CpuLowPageMetrics contiguous_pages = cpu_low_page_metrics(
        layout, storage, contiguous_owner);

    CpuLowPageMetrics lpt_domain_pages;
    CpuLowPageMetrics contiguous_domain_pages;
    CpuLowPageMetrics hybrid_pages;
    CpuLowPageMetrics hybrid_domain_pages;
    int domains = 0;
    double hybrid_imbalance = 0.0;
    if (domain_size > 0) {
        domains = (workers + domain_size - 1) / domain_size;
        lpt_domain_pages = cpu_low_page_metrics(
            layout, storage, cpu_low_domain_owner(lpt_owner, domain_size));
        contiguous_domain_pages = cpu_low_page_metrics(
            layout, storage, cpu_low_domain_owner(contiguous_owner, domain_size));
        uint64_t hybrid_total = std::accumulate(
            domain_pool.sticky_worker_cells.begin(),
            domain_pool.sticky_worker_cells.end(), uint64_t(0));
        if (hybrid_total != exact_total
            || cpu_low_assigned_jobs(domain_pool.sticky_worker_jobs) != nonempty_jobs) {
            std::cerr << "cpu LOW domain schedule accounting mismatch\n";
            return 8;
        }
        hybrid_imbalance = avg > 0.0
            ? double(cpu_low_max_cells(domain_pool.sticky_worker_cells)) / avg : 0.0;
        std::vector<int> domain_owner = cpu_low_owner_from_pool(domain_pool, jobs);
        hybrid_pages = cpu_low_page_metrics(layout, storage, domain_owner);
        hybrid_domain_pages = cpu_low_page_metrics(
            layout, storage, cpu_low_domain_owner(domain_owner, domain_size));
    }

    size_t contiguous_active_workers = 0;
    for (uint64_t x : contiguous_pool.sticky_worker_cells)
        if (x) ++contiguous_active_workers;

    std::cout << std::setprecision(12)
              << "cpu_low_schedule_plan OK"
              << " n=" << n
              << " workers=" << workers
              << " jobs=" << nonempty_jobs
              << " total_cells=" << exact_total
              << " min_worker_cells=" << min_cells
              << " max_worker_cells=" << max_cells
              << " avg_worker_cells=" << avg
              << " imbalance=" << imbalance
              << " shared_pages_4k=" << lpt_pages.shared_4k
              << " cross_worker_pages_4k=" << lpt_pages.cross_4k
              << " cross_worker_of_shared_4k=" << lpt_pages.cross_of_shared_4k
              << " cross_worker_auth_page_fraction_4k=" << lpt_pages.cross_auth_4k
              << " shared_pages_2m=" << lpt_pages.shared_2m
              << " cross_worker_pages_2m=" << lpt_pages.cross_2m
              << " cross_worker_of_shared_2m=" << lpt_pages.cross_of_shared_2m
              << " cross_worker_auth_page_fraction_2m=" << lpt_pages.cross_auth_2m
              << " contiguous_active_workers=" << contiguous_active_workers
              << " contiguous_optimal_cap=" << contiguous_pool.contiguous_optimal_cap
              << " contiguous_min_worker_cells=" << contiguous_min_cells
              << " contiguous_max_worker_cells=" << contiguous_max_cells
              << " contiguous_imbalance=" << contiguous_imbalance
              << " contiguous_cross_worker_pages_4k=" << contiguous_pages.cross_4k
              << " contiguous_cross_worker_of_shared_4k=" << contiguous_pages.cross_of_shared_4k
              << " contiguous_cross_worker_auth_page_fraction_4k=" << contiguous_pages.cross_auth_4k
              << " contiguous_cross_worker_pages_2m=" << contiguous_pages.cross_2m
              << " contiguous_cross_worker_of_shared_2m=" << contiguous_pages.cross_of_shared_2m
              << " contiguous_cross_worker_auth_page_fraction_2m=" << contiguous_pages.cross_auth_2m
              << " domain_size=" << domain_size
              << " domains=" << domains
              << " cross_domain_pages_4k=" << lpt_domain_pages.cross_4k
              << " cross_domain_auth_page_fraction_4k=" << lpt_domain_pages.cross_auth_4k
              << " cross_domain_pages_2m=" << lpt_domain_pages.cross_2m
              << " cross_domain_auth_page_fraction_2m=" << lpt_domain_pages.cross_auth_2m
              << " contiguous_cross_domain_pages_4k=" << contiguous_domain_pages.cross_4k
              << " contiguous_cross_domain_auth_page_fraction_4k=" << contiguous_domain_pages.cross_auth_4k
              << " contiguous_cross_domain_pages_2m=" << contiguous_domain_pages.cross_2m
              << " contiguous_cross_domain_auth_page_fraction_2m=" << contiguous_domain_pages.cross_auth_2m
              << " hybrid_domain_active_domains=" << domain_pool.domain_active_domains
              << " hybrid_domain_optimal_per_worker_cap="
              << domain_pool.domain_normalized_cap
              << " hybrid_domain_min_worker_cells="
              << cpu_low_min_cells(domain_pool.sticky_worker_cells)
              << " hybrid_domain_max_worker_cells="
              << cpu_low_max_cells(domain_pool.sticky_worker_cells)
              << " hybrid_domain_imbalance=" << hybrid_imbalance
              << " hybrid_domain_cross_worker_pages_4k=" << hybrid_pages.cross_4k
              << " hybrid_domain_cross_worker_auth_page_fraction_4k=" << hybrid_pages.cross_auth_4k
              << " hybrid_domain_cross_worker_pages_2m=" << hybrid_pages.cross_2m
              << " hybrid_domain_cross_worker_auth_page_fraction_2m=" << hybrid_pages.cross_auth_2m
              << " hybrid_domain_cross_domain_pages_4k=" << hybrid_domain_pages.cross_4k
              << " hybrid_domain_cross_domain_auth_page_fraction_4k=" << hybrid_domain_pages.cross_auth_4k
              << " hybrid_domain_cross_domain_pages_2m=" << hybrid_domain_pages.cross_2m
              << " hybrid_domain_cross_domain_auth_page_fraction_2m=" << hybrid_domain_pages.cross_auth_2m
              << " build_s=" << lpt_pool.schedule_build_s
              << " contiguous_build_s=" << contiguous_pool.schedule_build_s
              << " domain_build_s=" << domain_pool.schedule_build_s
              << '\n';

    if (dump_workers) {
        std::cout << "worker\tjobs\tcells\tfraction\tcontiguous_jobs\tcontiguous_cells\tcontiguous_fraction\thybrid_domain_jobs\thybrid_domain_cells\thybrid_domain_fraction\tdomain\n";
        for (int w = 0; w < workers; ++w) {
            uint64_t cells = lpt_pool.sticky_worker_cells[size_t(w)];
            uint64_t ccells = contiguous_pool.sticky_worker_cells[size_t(w)];
            uint64_t hcells = domain_size > 0
                ? domain_pool.sticky_worker_cells[size_t(w)] : 0;
            double fraction = exact_total ? double(cells) / double(exact_total) : 0.0;
            double cfraction = exact_total ? double(ccells) / double(exact_total) : 0.0;
            double hfraction = exact_total ? double(hcells) / double(exact_total) : 0.0;
            int domain = domain_size > 0 ? w / domain_size : -1;
            std::cout << w << '\t'
                      << lpt_pool.sticky_worker_jobs[size_t(w)].size() << '\t'
                      << cells << '\t' << fraction << '\t'
                      << contiguous_pool.sticky_worker_jobs[size_t(w)].size() << '\t'
                      << ccells << '\t' << cfraction << '\t'
                      << (domain_size > 0 ? domain_pool.sticky_worker_jobs[size_t(w)].size() : 0) << '\t'
                      << hcells << '\t' << hfraction << '\t'
                      << domain << '\n';
        }
    }
    return 0;
}
