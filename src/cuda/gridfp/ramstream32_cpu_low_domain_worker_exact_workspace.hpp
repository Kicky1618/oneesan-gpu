#pragma once

#include "ramstream32_cpu_low_domain_worker_dense_page.hpp"
#include "ramstream32_cpu_low_domain_worker_unique_coalesce.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

// Research-only v5.31 immutable workspace shared by exact worker-locality
// branches.  Everything here depends only on jobs/topology, never on a branch's
// current worker ownership, so direct/hybrid searches can reuse it safely.

struct CpuLowWorkerExactWorkspace {
    const std::vector<CpuLowJob>* source_jobs = nullptr;
    const CpuLowSparseHost* source_sparse = nullptr;
    std::vector<CpuLowStaticJobCost> ordered;
    std::vector<size_t> ordered_pos;
    CpuLowDomainPageMaskIndex mask_index;
    CpuLowWorkerDensePageIndex dense;
    std::vector<uint32_t> transition_weight;
    double mask_index_build_s = 0.0;
    double dense_index_build_s = 0.0;
    double transition_build_s = 0.0;
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
    ws.build_s = ram_seconds_since(t0);
    return ws;
}

static void cpu_low_validate_worker_exact_workspace(
    const CpuLowWorkerExactWorkspace& ws,
    const std::vector<CpuLowJob>& jobs,
    const CpuLowSparseHost& sparse
) {
    if (ws.source_jobs != &jobs || ws.source_sparse != &sparse
        || ws.ordered_pos.size() != jobs.size()
        || ws.dense.boundary.size() != ws.ordered.size() + 1
        || ws.transition_weight.size() != ws.ordered.size() + 1) {
        std::cerr << "cpu LOW exact workspace provenance mismatch\n";
        std::exit(240);
    }
}
