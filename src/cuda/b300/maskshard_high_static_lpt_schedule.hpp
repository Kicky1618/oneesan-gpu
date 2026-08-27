#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
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

#ifdef MASKSHARD_HIGH_CAP_LPT_SCHEDULE
#ifndef MASKSHARD_HIGH_CUDA_GRAPH
#error "HIGH cap LPT currently targets the v0.72 CUDA-Graph execution path"
#endif
#ifndef MASKSHARD_ROW_DEPTH_FBLOCK_IO
#error "HIGH cap LPT requires a host row-depth hook"
#endif
#ifndef MASKSHARD_ROW_DEPTH_ORBIT_COMPACT
#error "HIGH cap LPT requires exact compact HIGH orbit counts"
#endif
#ifndef MASKSHARD_HIGH_CLOSURE_ROW_DEPTH_COMPACT_LAUNCH
#error "HIGH cap LPT requires exact HIGH closure launch counts"
#endif
#ifndef MASKSHARD_HIGH_CLOSURE_LAUNCH_CLASS_CACHE
#error "HIGH cap LPT capture accounting requires popcount-class closure geometry"
#endif
static_assert((TARGET_W & 1) == 0,
              "HIGH cap LPT currently assumes the even-width graph cap model");

static std::atomic<int> G_MS_HIGH_CAP_LPT_ROW{-1};

static int maskshard_high_cap_lpt_popcount(std::uint32_t mask) {
    int n = 0;
    while (mask) {
        mask &= mask - 1;
        ++n;
    }
    return n;
}

static MaskShardHighStaticLptSchedule maskshard_build_high_lpt_from_weights(
    const std::vector<std::uint64_t>& weight, int ngpu
) {
    if (ngpu < 1) {
        std::cerr << "HIGH cap LPT requires at least one GPU\n";
        std::exit(366);
    }
    std::vector<std::size_t> order(weight.size());
    for (std::size_t q = 0; q < order.size(); ++q) order[q] = q;
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return weight[a] != weight[b] ? weight[a] > weight[b] : a < b;
    });

    MaskShardHighStaticLptSchedule out;
    out.jobs_by_gpu.resize(std::size_t(ngpu));
    out.work_by_gpu.assign(std::size_t(ngpu), 0);
    for (std::size_t q : order) {
        int best = 0;
        for (int d = 1; d < ngpu; ++d) {
            if (out.work_by_gpu[std::size_t(d)]
                < out.work_by_gpu[std::size_t(best)]) {
                best = d;
            }
        }
        const std::uint64_t w = weight[q];
        std::uint64_t& load = out.work_by_gpu[std::size_t(best)];
        if (w > std::numeric_limits<std::uint64_t>::max() - load
            || w > std::numeric_limits<std::uint64_t>::max() - out.total_work) {
            std::cerr << "HIGH cap LPT work counter overflow q=" << q << '\n';
            std::exit(367);
        }
        out.jobs_by_gpu[std::size_t(best)].push_back(q);
        load += w;
        out.total_work += w;
    }
    if (!out.work_by_gpu.empty()) {
        const auto mm = std::minmax_element(
            out.work_by_gpu.begin(), out.work_by_gpu.end());
        out.min_work = *mm.first;
        out.max_work = *mm.second;
    }
    return out;
}

template<class Job>
struct MaskShardHighCapLptState {
    static constexpr int FULL_CAP = TARGET_W / 2;

    const std::vector<Job>* jobs = nullptr;
    int ngpu = 0;
    MaskShardHighStaticLptSchedule baseline;
    std::vector<MaskShardHighStaticLptSchedule> by_cap;
    bool prepared = false;
    std::uint64_t baseline_capture_classes = 0;
    std::uint64_t cap_capture_classes = 0;
    std::uint64_t baseline_row_peak_work = 0;
    std::uint64_t cap_row_peak_work = 0;

    static void checked_add(std::uint64_t& dst, std::uint64_t x) {
        if (x > std::numeric_limits<std::uint64_t>::max() - dst) {
            std::cerr << "HIGH cap LPT model-work overflow\n";
            std::exit(368);
        }
        dst += x;
    }

    static std::uint64_t class_count(
        const MaskShardHighStaticLptSchedule& plan,
        const std::vector<Job>& jobs
    ) {
        std::uint64_t total = 0;
        for (const auto& gpu_jobs : plan.jobs_by_gpu) {
            bool seen[LOW_LUT_K + 1]{};
            for (std::size_t q : gpu_jobs) {
                const int pc = maskshard_high_cap_lpt_popcount(jobs[q].low_mask);
                if (pc < 0 || pc > LOW_LUT_K) {
                    std::cerr << "HIGH cap LPT invalid popcount " << pc << '\n';
                    std::exit(369);
                }
                if (!seen[pc]) {
                    seen[pc] = true;
                    ++total;
                }
            }
        }
        return total;
    }

    static std::uint64_t peak_on_plan(
        const MaskShardHighStaticLptSchedule& plan,
        const std::vector<std::uint64_t>& weight
    ) {
        std::uint64_t peak = 0;
        for (const auto& gpu_jobs : plan.jobs_by_gpu) {
            std::uint64_t load = 0;
            for (std::size_t q : gpu_jobs) checked_add(load, weight[q]);
            peak = std::max(peak, load);
        }
        return peak;
    }

    void prepare() {
        if (prepared) return;
        if (!jobs || ngpu < 1) {
            std::cerr << "HIGH cap LPT invalid lazy state\n";
            std::exit(370);
        }
        auto& orbit = maskshard_row_depth_orbit_compact_cache();
        if (!orbit.built) {
            std::cerr << "HIGH cap LPT orbit count cache not prepared\n";
            std::exit(371);
        }
        auto& closure = maskshard_highclosure_rowdepth_compact_launch_cache();
        if (!closure.built) {
            std::cerr << "HIGH cap LPT closure launch cache not prepared\n";
            std::exit(372);
        }

        by_cap.resize(FULL_CAP);
        baseline_capture_classes =
            std::uint64_t(FULL_CAP) * class_count(baseline, *jobs);
        cap_capture_classes = 0;
        baseline_row_peak_work = 0;
        cap_row_peak_work = 0;

        std::vector<std::uint64_t> weight(jobs->size(), 0);
        for (int cap = 1; cap <= FULL_CAP; ++cap) {
            for (std::size_t q = 0; q < jobs->size(); ++q) {
                const Job& job = (*jobs)[q];
                std::array<Code, HIGH_LUT_K + 3> orbit_prefix{};
                std::array<std::uint16_t, HIGH_LUT_K + 2> orbit_low_count{};
                const Code orbit_tasks = orbit.make_job_plan(
                    job.low_mask, cap, orbit_prefix, orbit_low_count);

                std::uint64_t work = 0;
                // Logical lane iterations. HIGH main I/O launches gather+scatter;
                // lazy BLOCKED init removes gather, leaving scatter only.
                checked_add(work, 2ULL * std::uint64_t(job.main_n));
#ifdef MASKSHARD_LAZY_ZERO_BLOCK_INIT
                checked_add(work, std::uint64_t(job.block_n));
#else
                checked_add(work, 2ULL * std::uint64_t(job.block_n));
#endif
                if (std::uint64_t(orbit_tasks)
                    > std::numeric_limits<std::uint64_t>::max()
                          / std::uint64_t(HIGH_LUT_K)) {
                    std::cerr << "HIGH cap LPT orbit model overflow mask="
                              << job.low_mask << " cap=" << cap << '\n';
                    std::exit(373);
                }
                checked_add(work,
                    std::uint64_t(orbit_tasks) * std::uint64_t(HIGH_LUT_K));
                for (int pi = 0; pi < HIGH_LUT_K; ++pi) {
                    const std::uint64_t warp_tasks =
                        maskshard_highclosure_rowdepth_compact_launch_tasks(
                            job.low_mask, pi, cap);
                    if (warp_tasks > std::numeric_limits<std::uint64_t>::max() / 32ULL) {
                        std::cerr << "HIGH cap LPT closure model overflow mask="
                                  << job.low_mask << " cap=" << cap
                                  << " pi=" << pi << '\n';
                        std::exit(374);
                    }
                    checked_add(work, warp_tasks * 32ULL);
                }
                weight[q] = work;
            }

            auto& plan = by_cap[std::size_t(cap - 1)];
            plan = maskshard_build_high_lpt_from_weights(weight, ngpu);
            cap_capture_classes += class_count(plan, *jobs);

            const std::uint64_t baseline_peak = peak_on_plan(baseline, weight);
            const std::uint64_t cap_peak = plan.max_work;
            const std::uint64_t row_count = cap < FULL_CAP
                ? 1ULL : std::uint64_t(TARGET_W - FULL_CAP + 1);
            if (baseline_peak > std::numeric_limits<std::uint64_t>::max() / row_count
                || cap_peak > std::numeric_limits<std::uint64_t>::max() / row_count) {
                std::cerr << "HIGH cap LPT row-weighted peak overflow cap=" << cap << '\n';
                std::exit(375);
            }
            checked_add(baseline_row_peak_work, baseline_peak * row_count);
            checked_add(cap_row_peak_work, cap_peak * row_count);

            std::cerr << "fullorbit-batch HIGH cap LPT cap=" << cap
                      << " total_work=" << plan.total_work
                      << " min_gpu_work=" << plan.min_work
                      << " max_gpu_work=" << plan.max_work
                      << " baseline_max_gpu_work=" << baseline_peak
                      << " capture_classes=" << class_count(plan, *jobs) << '\n';
        }

        prepared = true;
        const double reduction = baseline_row_peak_work
            ? 100.0 * (double(baseline_row_peak_work) - double(cap_row_peak_work))
                / double(baseline_row_peak_work)
            : 0.0;
        std::cerr << "fullorbit-batch HIGH cap LPT row_peak_baseline="
                  << baseline_row_peak_work
                  << " row_peak_cap_lpt=" << cap_row_peak_work
                  << " reduction_pct=" << reduction
                  << " graph_classes_baseline=" << baseline_capture_classes
                  << " graph_classes_cap_lpt=" << cap_capture_classes
                  << " graph_class_delta="
                  << (std::int64_t(cap_capture_classes)
                      - std::int64_t(baseline_capture_classes)) << '\n';
    }
};

template<class Job>
struct MaskShardHighCapLptJobsProxy {
    std::shared_ptr<MaskShardHighCapLptState<Job>> state;

    const std::vector<std::size_t>& operator[](std::size_t gpu) const {
        if (!state || !state->prepared) {
            std::cerr << "HIGH cap LPT schedule used before prepare\n";
            std::exit(376);
        }
        const int row = G_MS_HIGH_CAP_LPT_ROW.load(std::memory_order_acquire);
        if (row < 0 || row >= TARGET_W) {
            std::cerr << "HIGH cap LPT invalid current row " << row << '\n';
            std::exit(377);
        }
        if (gpu >= std::size_t(state->ngpu)) {
            std::cerr << "HIGH cap LPT invalid GPU " << gpu << '\n';
            std::exit(378);
        }
        const int cap = std::min(row + 1, MaskShardHighCapLptState<Job>::FULL_CAP);
        return state->by_cap[std::size_t(cap - 1)].jobs_by_gpu[gpu];
    }
};

template<class Job>
struct MaskShardHighCapLptSchedule {
    std::shared_ptr<MaskShardHighCapLptState<Job>> state;
    MaskShardHighCapLptJobsProxy<Job> jobs_by_gpu;
    std::uint64_t total_work = 0;
    std::uint64_t min_work = 0;
    std::uint64_t max_work = 0;

    void prepare() const { state->prepare(); }
};

template<class Job>
static MaskShardHighCapLptSchedule<Job> maskshard_build_high_cap_lpt_schedule(
    const std::vector<Job>& jobs, int ngpu
) {
    auto state = std::make_shared<MaskShardHighCapLptState<Job>>();
    state->jobs = &jobs;
    state->ngpu = ngpu;
    state->baseline = maskshard_build_high_static_lpt_schedule(jobs, ngpu);

    MaskShardHighCapLptSchedule<Job> out;
    out.state = state;
    out.jobs_by_gpu.state = state;
    // Preserve the v0.74 setup log until exact cap schedules are prepared by
    // the following HIGH-closure setup hook.
    out.total_work = state->baseline.total_work;
    out.min_work = state->baseline.min_work;
    out.max_work = state->baseline.max_work;
    return out;
}

template<class Schedule>
static void maskshard_prepare_highclosure_rowdepth_compact_cap_lpt(
    const HighDescHost& high_desc, int ngpu, const Schedule& schedule
) {
    maskshard_prepare_highclosure_rowdepth_compact_launch(high_desc, ngpu);
    schedule.prepare();
}

static void maskshard_set_row_depth_fblock_io_row_cap_lpt(int zero_based_row) {
    G_MS_HIGH_CAP_LPT_ROW.store(zero_based_row, std::memory_order_release);
    maskshard_set_row_depth_exact_io_row(zero_based_row);
}

// The shared v0.74 host remains untouched: redirect its setup-time schedule
// builder, closure prepare hook, and row-depth setter to the cap-aware layer.
#define maskshard_build_high_static_lpt_schedule \
        maskshard_build_high_cap_lpt_schedule
#ifdef maskshard_prepare_highclosure_rowdepth_compact
#undef maskshard_prepare_highclosure_rowdepth_compact
#endif
#define maskshard_prepare_highclosure_rowdepth_compact(high_desc, ngpu) \
        maskshard_prepare_highclosure_rowdepth_compact_cap_lpt( \
            high_desc, ngpu, high_schedule)
#ifdef maskshard_set_row_depth_fblock_io_row
#undef maskshard_set_row_depth_fblock_io_row
#endif
#define maskshard_set_row_depth_fblock_io_row \
        maskshard_set_row_depth_fblock_io_row_cap_lpt
#endif
