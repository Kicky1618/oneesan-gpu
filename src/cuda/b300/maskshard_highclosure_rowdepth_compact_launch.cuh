#pragma once

#ifndef MASKSHARD_HIGH_CLOSURE_ROW_DEPTH_COMPACT_LAUNCH
#error "maskshard_highclosure_rowdepth_compact_launch.cuh requires compact launch macro"
#endif
#ifndef MASKSHARD_HIGH_CLOSURE_ROW_DEPTH_COMPACT
#error "exact HIGH closure launch requires v0.22 compact task mapping"
#endif

// v0.23: host-only exact task-count table for the v0.22 kernel.  No new GPU
// metadata is required.  The table is indexed by fixed LOW occupancy mask,
// HIGH position and row-depth cap and stores the exact number of warp tasks
// before the 65,535-CTA cap is applied.
struct MaskShardHighClosureRowDepthCompactLaunchCache {
    static constexpr int FULL_CAP = (TARGET_W + 1) / 2;
    static constexpr int CAP_STRIDE = FULL_CAP + 1;
    static constexpr std::uint32_t NMASK = 1u << LOW_LUT_K;

    std::vector<std::uint32_t> task_count;
    bool built = false;

    static std::size_t index(std::uint32_t mask, int pi, int cap) {
        return (std::size_t(mask) * HIGH_LUT_K + std::size_t(pi)) * CAP_STRIDE
             + std::size_t(cap);
    }

    void build() {
        if (built) return;
        auto& hc = maskshard_highclosure_rowdepth_compact_cache();
        auto& low = maskshard_row_depth_orbit_compact_cache();
        if (!hc.built || !low.built) {
            std::cerr << "HIGH closure exact launch requires prepared compact metadata\n";
            std::exit(290);
        }

        task_count.assign(
            std::size_t(NMASK) * HIGH_LUT_K * CAP_STRIDE, 0u);
        for (std::uint32_t mask = 0; mask < NMASK; ++mask) {
            const auto blocks = make_factor_main_blocks(true, mask);
            for (int cap = 1; cap <= FULL_CAP; ++cap) {
                std::vector<std::uint16_t> low_active(blocks.size(), 0u);
                for (std::size_t bid = 0; bid < blocks.size(); ++bid) {
                    const FBlock& b = blocks[bid];
                    if (!b.stride) continue;
                    low_active[bid] = low.low_count[
                        MaskShardRowDepthOrbitCompactCache::low_count_index(
                            mask, int(b.hs), cap)];
                }
                for (int pi = 0; pi < HIGH_LUT_K; ++pi) {
                    std::uint64_t tasks = 0;
                    for (std::size_t bid = 0; bid < blocks.size(); ++bid) {
                        const FBlock& b = blocks[bid];
                        const std::uint32_t lc = low_active[bid];
                        if (!b.stride || !lc) continue;
                        const std::uint32_t rows = hc.active_count[
                            MaskShardHighClosureRowDepthCompactCache::count_index(
                                pi, int(bid), cap)];
                        if (!rows) continue;
#ifdef MASKSHARD_HIGH_CLOSURE_ROWPACK_THRESHOLD
                        const bool pack = b.stride
                            < std::uint32_t(MASKSHARD_HIGH_CLOSURE_ROWPACK_THRESHOLD);
#else
                        const bool pack = true;
#endif
                        tasks += pack
                            ? (std::uint64_t(rows) * lc + 31ULL) >> 5
                            : std::uint64_t(rows);
                    }
                    if (tasks > 0xffffffffULL) {
                        std::cerr << "HIGH closure exact launch task overflow mask="
                                  << mask << " pi=" << pi << " cap=" << cap
                                  << " tasks=" << tasks << '\n';
                        std::exit(291);
                    }
                    task_count[index(mask, pi, cap)] = std::uint32_t(tasks);
                }
            }
        }
        built = true;
        std::cerr << "HIGH closure exact launch host table entries="
                  << task_count.size() << " mib="
                  << double(task_count.size() * sizeof(std::uint32_t))
                       / double(1ULL << 20) << '\n';
    }
};

static MaskShardHighClosureRowDepthCompactLaunchCache&
maskshard_highclosure_rowdepth_compact_launch_cache() {
    static MaskShardHighClosureRowDepthCompactLaunchCache cache;
    return cache;
}

static void maskshard_prepare_highclosure_rowdepth_compact_launch(
    const HighDescHost& high_desc, int ngpu
) {
    maskshard_prepare_highclosure_rowdepth_compact(high_desc, ngpu);
    maskshard_highclosure_rowdepth_compact_launch_cache().build();
}

static std::uint32_t maskshard_highclosure_rowdepth_compact_launch_tasks(
    std::uint32_t mask, int pi, int cap
) {
    auto& cache = maskshard_highclosure_rowdepth_compact_launch_cache();
    if (!cache.built) {
        std::cerr << "HIGH closure exact launch table not prepared\n";
        std::exit(292);
    }
    cap = std::max(1, std::min(cap, cache.FULL_CAP));
    return cache.task_count[cache.index(mask, pi, cap)];
}

// Shared main calls this setup hook after HIGH jobs are built. Replace only the
// call site seen after this header; the original v0.22 function definition above
// remains available to the wrapper.
#define maskshard_prepare_highclosure_rowdepth_compact \
        maskshard_prepare_highclosure_rowdepth_compact_launch
