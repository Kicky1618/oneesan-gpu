#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <unordered_set>
#include <utility>
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

struct CpuLowOrderedEntry {
    uint32_t mask = 0;
    uint64_t cells = 0;
};

struct CpuLowContiguousPlan {
    std::vector<int> owner;
    std::vector<uint64_t> worker_cells;
    uint64_t min_cells = 0;
    uint64_t max_cells = 0;
    double imbalance = 0.0;
    uint64_t optimal_cap = 0;
    size_t active_workers = 0;
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

            uint32_t row0 = storage.high_mask_begin[size_t(mask) * StorageFactorHost::S + sb.he];
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

static size_t cpu_low_ordered_segments_needed(
    const std::vector<CpuLowOrderedEntry>& entries, uint64_t cap
) {
    if (entries.empty()) return 0;
    size_t segments = 1;
    uint64_t acc = 0;
    for (const auto& e : entries) {
        if (e.cells > cap) return std::numeric_limits<size_t>::max();
        if (acc && acc > cap - e.cells) {
            ++segments;
            acc = 0;
        }
        acc += e.cells;
    }
    return segments;
}

static uint64_t cpu_low_segment_cells(
    const std::vector<CpuLowOrderedEntry>& entries,
    const std::pair<size_t,size_t>& seg
) {
    uint64_t z = 0;
    for (size_t i = seg.first; i < seg.second; ++i) z += entries[i].cells;
    return z;
}

static CpuLowContiguousPlan cpu_low_build_optimal_contiguous_plan(
    const std::vector<CpuLowOrderedEntry>& entries, int workers,
    uint64_t exact_total
) {
    CpuLowContiguousPlan out;
    out.owner.assign(size_t(1) << HIGH_LUT_K, -1);
    out.worker_cells.assign(size_t(workers), 0);
    if (entries.empty()) return out;

    size_t target_segments = std::min<size_t>(size_t(workers), entries.size());
    uint64_t lo = 0;
    for (const auto& e : entries) lo = std::max(lo, e.cells);
    uint64_t hi = exact_total;
    while (lo < hi) {
        uint64_t mid = lo + (hi - lo) / 2;
        if (cpu_low_ordered_segments_needed(entries, mid) <= target_segments)
            hi = mid;
        else
            lo = mid + 1;
    }
    out.optimal_cap = lo;

    std::vector<std::pair<size_t,size_t>> segs;
    size_t begin = 0;
    uint64_t acc = 0;
    for (size_t i = 0; i < entries.size(); ++i) {
        uint64_t w = entries[i].cells;
        if (acc && acc > out.optimal_cap - w) {
            segs.push_back({begin, i});
            begin = i;
            acc = 0;
        }
        acc += w;
    }
    segs.push_back({begin, entries.size()});

    // The optimal cap may need fewer than target_segments. Split existing
    // segments until every available worker owns one contiguous run. Splitting
    // cannot increase the max load, so the min-max optimum is preserved.
    while (segs.size() < target_segments) {
        size_t best_seg = size_t(-1);
        uint64_t best_cells = 0;
        for (size_t s = 0; s < segs.size(); ++s) {
            if (segs[s].second - segs[s].first <= 1) continue;
            uint64_t cells = cpu_low_segment_cells(entries, segs[s]);
            if (best_seg == size_t(-1) || cells > best_cells) {
                best_seg = s;
                best_cells = cells;
            }
        }
        if (best_seg == size_t(-1)) {
            std::cerr << "cpu LOW contiguous split reconstruction failed\n";
            std::exit(6);
        }

        auto seg = segs[best_seg];
        uint64_t prefix = 0;
        uint64_t best_delta = std::numeric_limits<uint64_t>::max();
        size_t split = seg.first + 1;
        for (size_t i = seg.first; i + 1 < seg.second; ++i) {
            prefix += entries[i].cells;
            uint64_t right = best_cells - prefix;
            uint64_t delta = prefix > right ? prefix - right : right - prefix;
            if (delta < best_delta) {
                best_delta = delta;
                split = i + 1;
            }
        }
        segs[best_seg] = {seg.first, split};
        segs.insert(segs.begin() + best_seg + 1, {split, seg.second});
    }

    out.active_workers = segs.size();
    for (size_t w = 0; w < segs.size(); ++w) {
        uint64_t cells = 0;
        for (size_t i = segs[w].first; i < segs[w].second; ++i) {
            uint32_t mask = entries[i].mask;
            if (out.owner[mask] >= 0) {
                std::cerr << "cpu LOW contiguous duplicate owner mask=" << mask << '\n';
                std::exit(7);
            }
            out.owner[mask] = int(w);
            cells += entries[i].cells;
        }
        out.worker_cells[w] = cells;
        if (cells > out.optimal_cap) {
            std::cerr << "cpu LOW contiguous cap violation worker=" << w
                      << " cells=" << cells << " cap=" << out.optimal_cap << '\n';
            std::exit(8);
        }
    }

    out.min_cells = out.worker_cells.empty() ? 0 : out.worker_cells[0];
    out.max_cells = 0;
    for (uint64_t x : out.worker_cells) {
        out.min_cells = std::min(out.min_cells, x);
        out.max_cells = std::max(out.max_cells, x);
    }
    double avg = workers ? double(exact_total) / workers : 0.0;
    out.imbalance = avg > 0.0 ? double(out.max_cells) / avg : 0.0;
    return out;
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
            if (domain_size <= 0) {
                std::cerr << "--domain-size requires a positive integer\n";
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
    CpuLowSparsePersistentPool pool(workers, CPU_LOW_SCHEDULE_STICKY);
    pool.prepare_sticky_schedule(jobs, sparse);

    size_t nonempty_jobs = 0;
    uint64_t exact_total = 0;
    std::vector<CpuLowOrderedEntry> ordered;
    ordered.reserve(jobs.size());
    for (const auto& job : jobs) {
        if (!job.main_size && !job.block_size) continue;
        uint64_t cells = cpu_low_sparse_job_cells(job, sparse);
        ++nonempty_jobs;
        exact_total += cells;
        ordered.push_back({job.mask, cells});
    }
    std::sort(ordered.begin(), ordered.end(), [](const auto& a, const auto& b) {
        return a.mask < b.mask;
    });

    uint64_t assigned_total = std::accumulate(
        pool.sticky_worker_cells.begin(), pool.sticky_worker_cells.end(), uint64_t(0));
    size_t assigned_jobs = 0;
    for (const auto& v : pool.sticky_worker_jobs) assigned_jobs += v.size();
    if (assigned_total != exact_total || assigned_jobs != nonempty_jobs) {
        std::cerr << "cpu LOW sticky plan accounting mismatch jobs="
                  << assigned_jobs << '/' << nonempty_jobs
                  << " cells=" << assigned_total << '/' << exact_total << '\n';
        return 2;
    }

    uint64_t min_cells = pool.sticky_worker_cells.empty() ? 0 : pool.sticky_worker_cells[0];
    uint64_t max_cells = 0;
    for (uint64_t x : pool.sticky_worker_cells) {
        min_cells = std::min(min_cells, x);
        max_cells = std::max(max_cells, x);
    }
    double avg = workers ? double(exact_total) / workers : 0.0;
    double imbalance = avg > 0.0 ? double(max_cells) / avg : 0.0;

    std::vector<int> owner(size_t(1) << HIGH_LUT_K, -1);
    for (int w = 0; w < workers; ++w) {
        for (size_t q : pool.sticky_worker_jobs[size_t(w)]) {
            uint32_t mask = jobs[q].mask;
            if (owner[mask] >= 0) {
                std::cerr << "cpu LOW sticky duplicate mask owner mask=" << mask << '\n';
                return 5;
            }
            owner[mask] = w;
        }
    }

    CpuLowPageMetrics lpt_pages = cpu_low_page_metrics(layout, storage, owner);
    CpuLowContiguousPlan contiguous = cpu_low_build_optimal_contiguous_plan(
        ordered, workers, exact_total);
    CpuLowPageMetrics contiguous_pages = cpu_low_page_metrics(
        layout, storage, contiguous.owner);

    CpuLowPageMetrics lpt_domain_pages;
    CpuLowPageMetrics contiguous_domain_pages;
    int domains = 0;
    if (domain_size > 0) {
        domains = (workers + domain_size - 1) / domain_size;
        lpt_domain_pages = cpu_low_page_metrics(
            layout, storage, cpu_low_domain_owner(owner, domain_size));
        contiguous_domain_pages = cpu_low_page_metrics(
            layout, storage, cpu_low_domain_owner(contiguous.owner, domain_size));
    }

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
              << " contiguous_active_workers=" << contiguous.active_workers
              << " contiguous_optimal_cap=" << contiguous.optimal_cap
              << " contiguous_min_worker_cells=" << contiguous.min_cells
              << " contiguous_max_worker_cells=" << contiguous.max_cells
              << " contiguous_imbalance=" << contiguous.imbalance
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
              << " build_s=" << pool.schedule_build_s
              << '\n';

    if (dump_workers) {
        std::cout << "worker\tjobs\tcells\tfraction\tcontiguous_cells\tcontiguous_fraction\tdomain\n";
        for (int w = 0; w < workers; ++w) {
            uint64_t cells = pool.sticky_worker_cells[size_t(w)];
            uint64_t ccells = contiguous.worker_cells[size_t(w)];
            double fraction = exact_total ? double(cells) / double(exact_total) : 0.0;
            double cfraction = exact_total ? double(ccells) / double(exact_total) : 0.0;
            int domain = domain_size > 0 ? w / domain_size : -1;
            std::cout << w << '\t'
                      << pool.sticky_worker_jobs[size_t(w)].size() << '\t'
                      << cells << '\t' << fraction << '\t'
                      << ccells << '\t' << cfraction << '\t'
                      << domain << '\n';
        }
    }
    return 0;
}
