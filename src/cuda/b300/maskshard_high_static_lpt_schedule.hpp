#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

struct MaskShardHighStaticLptSchedule {
    std::vector<std::vector<std::size_t>> jobs_by_gpu;
    std::vector<std::uint64_t> work_by_gpu;
    std::uint64_t total_work = 0;
    std::uint64_t min_work = 0;
    std::uint64_t max_work = 0;
};

template<class Job>
static MaskShardHighStaticLptSchedule maskshard_build_high_static_lpt_schedule(
    const std::vector<Job>& jobs, int ngpu
) {
    if (ngpu < 1) {
        std::cerr << "HIGH static LPT requires at least one GPU\n";
        std::exit(360);
    }
    for (std::size_t q = 1; q < jobs.size(); ++q) {
        if (jobs[q].work > jobs[q - 1].work) {
            std::cerr << "HIGH static LPT requires nonincreasing job.work order q="
                      << q << '\n';
            std::exit(361);
        }
    }

    MaskShardHighStaticLptSchedule out;
    out.jobs_by_gpu.resize(std::size_t(ngpu));
    out.work_by_gpu.assign(std::size_t(ngpu), 0);
    std::vector<std::uint8_t> seen(jobs.size(), 0);

    for (std::size_t q = 0; q < jobs.size(); ++q) {
        int best = 0;
        for (int d = 1; d < ngpu; ++d) {
            if (out.work_by_gpu[std::size_t(d)]
                < out.work_by_gpu[std::size_t(best)]) {
                best = d;
            }
        }
        const std::uint64_t w = std::uint64_t(jobs[q].work);
        std::uint64_t& load = out.work_by_gpu[std::size_t(best)];
        if (w > std::numeric_limits<std::uint64_t>::max() - load
            || w > std::numeric_limits<std::uint64_t>::max() - out.total_work) {
            std::cerr << "HIGH static LPT work counter overflow q=" << q << '\n';
            std::exit(362);
        }
        out.jobs_by_gpu[std::size_t(best)].push_back(q);
        load += w;
        out.total_work += w;
        if (seen[q]++) {
            std::cerr << "HIGH static LPT duplicate job q=" << q << '\n';
            std::exit(363);
        }
    }

    std::size_t assigned = 0;
    for (const auto& v : out.jobs_by_gpu) assigned += v.size();
    if (assigned != jobs.size()) {
        std::cerr << "HIGH static LPT assignment count mismatch got=" << assigned
                  << " expected=" << jobs.size() << '\n';
        std::exit(364);
    }
    for (std::size_t q = 0; q < seen.size(); ++q) {
        if (seen[q] != 1) {
            std::cerr << "HIGH static LPT missing job q=" << q << '\n';
            std::exit(365);
        }
    }

    if (!out.work_by_gpu.empty()) {
        const auto mm = std::minmax_element(
            out.work_by_gpu.begin(), out.work_by_gpu.end());
        out.min_work = *mm.first;
        out.max_work = *mm.second;
    }
    return out;
}
