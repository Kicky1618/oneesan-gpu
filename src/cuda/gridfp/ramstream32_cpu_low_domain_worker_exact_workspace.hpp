#pragma once

#include "ramstream32_cpu_low_domain_worker_dense_page.hpp"
#include "ramstream32_cpu_low_domain_worker_unique_coalesce.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

// Research-only v5.31 immutable workspace shared by exact worker-locality
// branches. Everything here depends only on jobs/topology, never on a branch's
// current worker ownership, so direct/hybrid searches can reuse it safely.
//
// The expensive structural audit is performed exactly once at workspace build
// time. Per-search validation remains O(1) and only checks provenance/sizes.

struct CpuLowWorkerExactWorkspace {
    const std::vector<CpuLowJob>* source_jobs = nullptr;
    const CpuLowSparseHost* source_sparse = nullptr;
    std::vector<CpuLowStaticJobCost> ordered;
    std::vector<size_t> ordered_pos;
    CpuLowDomainPageMaskIndex mask_index;
    CpuLowWorkerDensePageIndex dense;
    std::vector<uint32_t> transition_weight;
    uint64_t audited_jobs = 0;
    uint64_t audited_cells = 0;
    bool structural_audit_ok = false;
    double mask_index_build_s = 0.0;
    double dense_index_build_s = 0.0;
    double transition_build_s = 0.0;
    double audit_s = 0.0;
    double build_s = 0.0;

    size_t bytes() const {
        return ordered.size() * sizeof(CpuLowStaticJobCost)
            + ordered_pos.size() * sizeof(size_t)
            + mask_index.first_nonempty.size() * sizeof(uint32_t)
            + mask_index.next_nonempty.size() * sizeof(uint32_t)
            + dense.bytes()
            + transition_weight.size() * sizeof(uint32_t);
    }
};

static bool cpu_low_exact_dense_ids_valid(
    const std::vector<uint32_t>& ids,
    size_t universe_size
) {
    if (!std::is_sorted(ids.begin(), ids.end())) return false;
    if (std::adjacent_find(ids.begin(), ids.end()) != ids.end()) return false;
    for (uint32_t id : ids)
        if (size_t(id) >= universe_size) return false;
    return true;
}

static void cpu_low_audit_worker_exact_workspace(
    CpuLowWorkerExactWorkspace& ws,
    const std::vector<CpuLowJob>& jobs,
    const CpuLowSparseHost& sparse
) {
    auto t0 = std::chrono::steady_clock::now();
    if (ws.source_jobs != &jobs || ws.source_sparse != &sparse
        || ws.ordered_pos.size() != jobs.size()
        || ws.dense.boundary.size() != ws.ordered.size() + 1
        || ws.transition_weight.size() != ws.ordered.size() + 1) {
        std::cerr << "cpu LOW exact workspace structural size mismatch\n";
        std::exit(309);
    }

    uint64_t total_cells = 0;
    size_t nonempty_jobs = 0;
    for (size_t i = 0; i < jobs.size(); ++i) {
        bool nonempty = jobs[i].main_size || jobs[i].block_size;
        size_t pos = ws.ordered_pos[i];
        if (!nonempty) {
            if (pos != size_t(-1)) {
                std::cerr << "cpu LOW exact workspace empty job was indexed\n";
                std::exit(310);
            }
            continue;
        }
        ++nonempty_jobs;
        if (pos == size_t(-1) || pos >= ws.ordered.size()) {
            std::cerr << "cpu LOW exact workspace nonempty job missing\n";
            std::exit(311);
        }
        const auto& x = ws.ordered[pos];
        uint64_t cells = cpu_low_sparse_job_cells(jobs[i], sparse);
        if (x.index != i || x.mask != jobs[i].mask || x.cells != cells) {
            std::cerr << "cpu LOW exact workspace inverse job mismatch\n";
            std::exit(312);
        }
    }
    if (nonempty_jobs != ws.ordered.size()) {
        std::cerr << "cpu LOW exact workspace nonempty job count mismatch\n";
        std::exit(313);
    }

    for (size_t k = 0; k < ws.ordered.size(); ++k) {
        const auto& x = ws.ordered[k];
        if (x.index >= jobs.size() || ws.ordered_pos[x.index] != k
            || x.mask >= ws.mask_index.nmasks) {
            std::cerr << "cpu LOW exact workspace ordered provenance mismatch\n";
            std::exit(314);
        }
        if (k) {
            const auto& p = ws.ordered[k - 1];
            if (p.mask > x.mask || (p.mask == x.mask && p.index >= x.index)) {
                std::cerr << "cpu LOW exact workspace order is not strict\n";
                std::exit(315);
            }
        }
        if (x.cells > std::numeric_limits<uint64_t>::max() - total_cells) {
            std::cerr << "cpu LOW exact workspace cell sum overflow\n";
            std::exit(316);
        }
        total_cells += x.cells;
    }

    auto universe_valid = [](const std::vector<uint64_t>& u) {
        return std::is_sorted(u.begin(), u.end())
            && std::adjacent_find(u.begin(), u.end()) == u.end()
            && u.size() <= uint64_t(std::numeric_limits<uint32_t>::max()) + 1ull;
    };
    if (!universe_valid(ws.dense.universe_2m)
        || !universe_valid(ws.dense.universe_4k)) {
        std::cerr << "cpu LOW exact workspace dense universe invalid\n";
        std::exit(317);
    }
    if ((!ws.dense.boundary.empty()
            && (!ws.dense.boundary.front().pages_2m.empty()
                || !ws.dense.boundary.front().pages_4k.empty()
                || !ws.dense.boundary.back().pages_2m.empty()
                || !ws.dense.boundary.back().pages_4k.empty()))
        || (!ws.transition_weight.empty()
            && (ws.transition_weight.front() != 0
                || ws.transition_weight.back() != 0))) {
        std::cerr << "cpu LOW exact workspace endpoint sentinel mismatch\n";
        std::exit(318);
    }
    for (const auto& sig : ws.dense.boundary) {
        if (!cpu_low_exact_dense_ids_valid(sig.pages_2m, ws.dense.universe_2m.size())
            || !cpu_low_exact_dense_ids_valid(sig.pages_4k, ws.dense.universe_4k.size())) {
            std::cerr << "cpu LOW exact workspace dense boundary invalid\n";
            std::exit(319);
        }
    }

    ws.audited_jobs = nonempty_jobs;
    ws.audited_cells = total_cells;
    ws.structural_audit_ok = true;
    ws.audit_s = ram_seconds_since(t0);
}

static CpuLowWorkerExactWorkspace cpu_low_build_worker_exact_workspace(
    const std::vector<CpuLowJob>& jobs,
    const CpuLowSparseHost& sparse,
    const StorageFactorHost& storage,
    const StorageLayout& layout
) {
    CpuLowWorkerExactWorkspace ws;
    auto t0 = std::chrono::steady_clock::now();
    ws.source_jobs = &jobs;
    ws.source_sparse = &sparse;
    ws.ordered.reserve(jobs.size());
    ws.ordered_pos.assign(jobs.size(), size_t(-1));

    for (size_t i = 0; i < jobs.size(); ++i) {
        if (!jobs[i].main_size && !jobs[i].block_size) continue;
        ws.ordered.push_back({i, jobs[i].mask, cpu_low_sparse_job_cells(jobs[i], sparse)});
    }
    std::sort(ws.ordered.begin(), ws.ordered.end(), [](const auto& a, const auto& b) {
        if (a.mask != b.mask) return a.mask < b.mask;
        return a.index < b.index;
    });
    for (size_t k = 0; k < ws.ordered.size(); ++k) {
        size_t q = ws.ordered[k].index;
        if (q >= jobs.size() || ws.ordered_pos[q] != size_t(-1)) {
            std::cerr << "cpu LOW exact workspace ordered index mismatch\n";
            std::exit(239);
        }
        ws.ordered_pos[q] = k;
    }

    auto mt0 = std::chrono::steady_clock::now();
    ws.mask_index = cpu_low_build_domain_page_mask_index();
    ws.mask_index_build_s = ram_seconds_since(mt0);

    ws.dense = cpu_low_build_worker_dense_page_index(
        ws.ordered, layout, storage, ws.mask_index);
    ws.dense_index_build_s = ws.dense.build_s;

    auto tt0 = std::chrono::steady_clock::now();
    ws.transition_weight.assign(ws.ordered.size() + 1, 0);
    for (size_t boundary = 1; boundary < ws.ordered.size(); ++boundary) {
        ws.transition_weight[boundary] = cpu_low_worker_unique_transition_weight(
            layout, ws.mask_index, ws.ordered[boundary].mask);
    }
    ws.transition_build_s = ram_seconds_since(tt0);

    cpu_low_audit_worker_exact_workspace(ws, jobs, sparse);
    ws.build_s = ram_seconds_since(t0);
    return ws;
}

static void cpu_low_validate_worker_exact_workspace(
    const CpuLowWorkerExactWorkspace& ws,
    const std::vector<CpuLowJob>& jobs,
    const CpuLowSparseHost& sparse
) {
    if (ws.source_jobs != &jobs || ws.source_sparse != &sparse
        || !ws.structural_audit_ok
        || ws.audited_jobs != ws.ordered.size()
        || ws.ordered_pos.size() != jobs.size()
        || ws.dense.boundary.size() != ws.ordered.size() + 1
        || ws.transition_weight.size() != ws.ordered.size() + 1) {
        std::cerr << "cpu LOW exact workspace provenance mismatch\n";
        std::exit(240);
    }
}
