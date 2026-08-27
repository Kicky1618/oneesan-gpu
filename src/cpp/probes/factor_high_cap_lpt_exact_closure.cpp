#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <vector>

#define TARGET_W 28
#define LOW_LUT_K 14
#define HIGH_LUT_K 13
#define MASKSHARD_HIGH_CAP_LPT_SCHEDULE 1
#define MASKSHARD_HIGH_CAP_LPT_EXACT_CLOSURE_LANES 1
#define MASKSHARD_HIGH_CUDA_GRAPH 1
#define MASKSHARD_ROW_DEPTH_FBLOCK_IO 1
#define MASKSHARD_ROW_DEPTH_ORBIT_COMPACT 1
#define MASKSHARD_HIGH_CLOSURE_ROW_DEPTH_COMPACT 1
#define MASKSHARD_HIGH_CLOSURE_ROW_DEPTH_COMPACT_LAUNCH 1
#define MASKSHARD_HIGH_CLOSURE_LAUNCH_CLASS_CACHE 1
#define MASKSHARD_LAZY_ZERO_BLOCK_INIT 1
#define MASKSHARD_HIGH_CLOSURE_ROWPACK_THRESHOLD 29

using Code = std::uint64_t;
struct HighDescHost {};

struct FBlock {
    std::uint32_t stride = 0;
    std::uint32_t hs = 0;
};

static std::vector<FBlock> make_factor_main_blocks(bool, std::uint32_t) {
    return {{20u, 0u}, {80u, 1u}};
}

struct ProbeOrbitCache {
    bool built = true;
    Code make_job_plan(
        std::uint32_t,
        int,
        std::array<Code, HIGH_LUT_K + 3>& prefix,
        std::array<std::uint16_t, HIGH_LUT_K + 2>& low_count
    ) {
        prefix.fill(0);
        low_count.fill(0);
        low_count[0] = 17;
        low_count[1] = 65;
        prefix[1] = 17;
        prefix[2] = 82;
        for (std::size_t i = 3; i < prefix.size(); ++i) prefix[i] = 82;
        return 82;
    }
};

static ProbeOrbitCache& maskshard_row_depth_orbit_compact_cache() {
    static ProbeOrbitCache cache;
    return cache;
}

struct ProbeClosureCompactCache {
    static constexpr int FULL_CAP = (TARGET_W + 1) / 2;
    static constexpr int CAP_STRIDE = FULL_CAP + 1;
    static constexpr int BLOCK_STRIDE = 65;
    bool built = true;
    std::vector<std::uint32_t> active_count;

    ProbeClosureCompactCache()
        : active_count(std::size_t(HIGH_LUT_K) * BLOCK_STRIDE * CAP_STRIDE, 0u) {
        for (int pi = 0; pi < HIGH_LUT_K; ++pi) {
            for (int cap = 1; cap <= FULL_CAP; ++cap) {
                active_count[count_index(pi, 0, cap)] = 2;
                active_count[count_index(pi, 1, cap)] = 3;
            }
        }
    }

    static std::size_t count_index(int pi, int bid, int cap) {
        return (std::size_t(pi) * BLOCK_STRIDE + std::size_t(bid)) * CAP_STRIDE
             + std::size_t(cap);
    }
};

static ProbeClosureCompactCache& maskshard_highclosure_rowdepth_compact_cache() {
    static ProbeClosureCompactCache cache;
    return cache;
}

struct ProbeClosureLaunchCache { bool built = true; };
static ProbeClosureLaunchCache& maskshard_highclosure_rowdepth_compact_launch_cache() {
    static ProbeClosureLaunchCache cache;
    return cache;
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
    std::array<std::uint16_t, HIGH_LUT_K + 2> low_count{};
    low_count[0] = 17;
    low_count[1] = 65;
    const std::uint64_t exact =
        MaskShardHighCapLptState<Job>::exact_closure_lane_work(
            make_factor_main_blocks(true, 0), low_count, 1);
    const std::uint64_t legacy = std::uint64_t(HIGH_LUT_K) * 5ULL * 32ULL;
    // Per pi: packed = ceil(2*17/32)*32 = 64 lanes.
    // Unpacked = 3 rows * ceil(65/32) chunks * 32 = 288 lanes.
    const std::uint64_t expected = std::uint64_t(HIGH_LUT_K) * 352ULL;
    if (exact != expected || exact <= legacy) {
        std::cerr << "exact closure-lane model mismatch exact=" << exact
                  << " expected=" << expected << " legacy=" << legacy << '\n';
        return 1;
    }

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
    if (!high_schedule.state->prepared) {
        std::cerr << "exact closure-lane cap LPT setup did not prepare\n";
        return 2;
    }

    for (int row = 0; row < TARGET_W; ++row) {
        maskshard_set_row_depth_fblock_io_row(row);
        std::vector<std::uint8_t> seen(jobs.size(), 0);
        for (int d = 0; d < 8; ++d) {
            for (std::size_t q : high_schedule.jobs_by_gpu[std::size_t(d)]) {
                if (q >= jobs.size() || seen[q]++) {
                    std::cerr << "exact closure-lane duplicate/invalid assignment row="
                              << row << " gpu=" << d << " q=" << q << '\n';
                    return 3;
                }
            }
        }
        for (std::size_t q = 0; q < seen.size(); ++q) {
            if (seen[q] != 1) {
                std::cerr << "exact closure-lane missing assignment row="
                          << row << " q=" << q << '\n';
                return 4;
            }
        }
    }

    std::cout << "factor-high-cap-lpt-exact-closure OK exact=" << exact
              << " legacy=" << legacy
              << " ratio=" << double(exact) / double(legacy)
              << " caps=" << high_schedule.state->by_cap.size() << '\n';
    return 0;
}
