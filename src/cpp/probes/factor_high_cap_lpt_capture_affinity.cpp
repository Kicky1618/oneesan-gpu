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
#define MASKSHARD_HIGH_CAP_LPT_CAPTURE_AFFINITY 1
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
    // Synthetic tie case: plain LPT duplicates both popcount classes across
    // both GPUs. Affinity can swap only equal-load GPU identities and keep the
    // complete load multiset unchanged while reducing four graph classes to two.
    const std::vector<Job> tie_jobs = {
        {0u, 0, 0, 10}, // popcount 0
        {1u, 0, 0, 10}, // popcount 1
        {2u, 0, 0, 10}, // popcount 1
        {0u, 0, 0, 10}, // popcount 0
    };
    const std::vector<std::uint64_t> tie_weight(4, 10);
    const auto plain = maskshard_build_high_lpt_from_weights(tie_weight, 2);
    const auto affinity = maskshard_build_high_lpt_from_weights_affinity(
        tie_weight, tie_jobs, 2);
    if (!MaskShardHighCapLptState<Job>::same_load_multiset(plain, affinity)) {
        std::cerr << "capture affinity changed LPT load multiset\n";
        return 1;
    }
    const auto plain_classes =
        MaskShardHighCapLptState<Job>::class_count(plain, tie_jobs);
    const auto affinity_classes =
        MaskShardHighCapLptState<Job>::class_count(affinity, tie_jobs);
    if (plain_classes != 4 || affinity_classes != 2) {
        std::cerr << "capture affinity synthetic class mismatch plain="
                  << plain_classes << " affinity=" << affinity_classes << '\n';
        return 2;
    }

    // Exercise the real v0.77 cap-plan selection path as well.
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
        std::cerr << "capture-affinity cap LPT setup did not prepare\n";
        return 3;
    }
    if (high_schedule.state->cap_capture_classes
        > high_schedule.state->plain_cap_capture_classes) {
        std::cerr << "capture-affinity selected plan regressed graph classes\n";
        return 4;
    }

    for (int cap = 0; cap < MaskShardHighCapLptState<Job>::FULL_CAP; ++cap) {
        const auto& selected = high_schedule.state->by_cap[std::size_t(cap)];
        // prepare() asserts exact equality against the plain plan before selecting.
        if (selected.work_by_gpu.size() != 8) {
            std::cerr << "capture-affinity invalid selected GPU count cap=" << cap + 1 << '\n';
            return 5;
        }
    }

    std::cout << "factor-high-cap-lpt-capture-affinity OK plain_classes="
              << plain_classes << " affinity_classes=" << affinity_classes
              << " selected_caps=" << high_schedule.state->affinity_selected_caps
              << " cap_classes=" << high_schedule.state->cap_capture_classes
              << " plain_cap_classes="
              << high_schedule.state->plain_cap_capture_classes << '\n';
    return 0;
}
