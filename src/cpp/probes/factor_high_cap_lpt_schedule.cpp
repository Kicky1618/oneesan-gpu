#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <vector>

#define TARGET_W 28
#define LOW_LUT_K 14
#define HIGH_LUT_K 13
#define MASKSHARD_HIGH_CAP_LPT_SCHEDULE 1
#define MASKSHARD_HIGH_CUDA_GRAPH 1
#define MASKSHARD_ROW_DEPTH_FBLOCK_IO 1
#define MASKSHARD_ROW_DEPTH_ORBIT_COMPACT 1
#define MASKSHARD_HIGH_CLOSURE_ROW_DEPTH_COMPACT_LAUNCH 1
#define MASKSHARD_HIGH_CLOSURE_LAUNCH_CLASS_CACHE 1
#define MASKSHARD_LAZY_ZERO_BLOCK_INIT 1

using Code = std::uint64_t;
struct HighDescHost {};

struct ProbeOrbitCache {
    bool built = true;
    Code make_job_plan(
        std::uint32_t mask,
        int cap,
        std::array<Code, HIGH_LUT_K + 3>&,
        std::array<std::uint16_t, HIGH_LUT_K + 2>&
    ) {
        return Code((mask & 15u) + 1u) * Code(cap);
    }
};

static ProbeOrbitCache& maskshard_row_depth_orbit_compact_cache() {
    static ProbeOrbitCache cache;
    return cache;
}

struct ProbeClosureCache { bool built = true; };
static ProbeClosureCache& maskshard_highclosure_rowdepth_compact_launch_cache() {
    static ProbeClosureCache cache;
    return cache;
}

static std::uint32_t maskshard_highclosure_rowdepth_compact_launch_tasks(
    std::uint32_t mask, int pi, int cap
) {
    return ((mask + std::uint32_t(pi) + std::uint32_t(cap)) % 11u) + 1u;
}

static void maskshard_prepare_highclosure_rowdepth_compact_launch(
    const HighDescHost&, int
) {}
static void maskshard_set_row_depth_exact_io_row(int) {}
#define maskshard_prepare_highclosure_rowdepth_compact \
        maskshard_prepare_highclosure_rowdepth_compact_launch
#define maskshard_set_row_depth_fblock_io_row \
        maskshard_set_row_depth_exact_io_row

#include "../../cuda/b300/maskshard_high_static_lpt_schedule.hpp"

struct Job {
    std::uint32_t low_mask = 0;
    Code main_n = 0;
    Code block_n = 0;
    Code work = 0;
};

int main() {
    std::vector<Job> jobs;
    for (std::uint32_t mask = 0; mask < 64; ++mask) {
        const Code main_n = 1000 - mask;
        const Code block_n = 500 - mask / 2;
        jobs.push_back({mask, main_n, block_n, main_n + block_n});
    }
    std::sort(jobs.begin(), jobs.end(), [](const Job& a, const Job& b) {
        return a.work > b.work;
    });

    const auto high_schedule = maskshard_build_high_static_lpt_schedule(jobs, 8);
    HighDescHost high_desc;
    maskshard_prepare_highclosure_rowdepth_compact(high_desc, 8);
    if (!high_schedule.state->prepared
        || high_schedule.state->by_cap.size() != TARGET_W / 2) {
        std::cerr << "cap LPT setup did not prepare all caps\n";
        return 1;
    }

    for (int row = 0; row < TARGET_W; ++row) {
        maskshard_set_row_depth_fblock_io_row(row);
        std::vector<std::uint8_t> seen(jobs.size(), 0);
        std::size_t assigned = 0;
        for (int d = 0; d < 8; ++d) {
            for (std::size_t q : high_schedule.jobs_by_gpu[std::size_t(d)]) {
                if (q >= jobs.size() || seen[q]++) {
                    std::cerr << "cap LPT duplicate/invalid job row=" << row
                              << " gpu=" << d << " q=" << q << '\n';
                    return 2;
                }
                ++assigned;
            }
        }
        if (assigned != jobs.size()) {
            std::cerr << "cap LPT assignment count mismatch row=" << row
                      << " got=" << assigned << " expected=" << jobs.size() << '\n';
            return 3;
        }
    }

    if (high_schedule.state->baseline_capture_classes == 0
        || high_schedule.state->cap_capture_classes == 0
        || high_schedule.state->baseline_row_peak_work == 0
        || high_schedule.state->cap_row_peak_work == 0) {
        std::cerr << "cap LPT model counters were not populated\n";
        return 4;
    }

    std::cout << "factor-high-cap-lpt OK jobs=" << jobs.size()
              << " caps=" << high_schedule.state->by_cap.size()
              << " baseline_row_peak="
              << high_schedule.state->baseline_row_peak_work
              << " cap_row_peak=" << high_schedule.state->cap_row_peak_work
              << " baseline_graph_classes="
              << high_schedule.state->baseline_capture_classes
              << " cap_graph_classes="
              << high_schedule.state->cap_capture_classes << '\n';
    return 0;
}
