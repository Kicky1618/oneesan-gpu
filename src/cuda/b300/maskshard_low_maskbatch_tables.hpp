#pragma once

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#ifndef MASKSHARD_LOW_ORBIT_ROW_DEPTH_COMPACT
#error "LOW mask-batch tables require exact compact LOW orbit metadata"
#endif
#ifndef MASKSHARD_LOW_CLOSURE_PACKED_PREFIX
#error "LOW mask-batch tables require captured exact LOW closure metadata"
#endif

struct MaskShardLowMaskBatchTablesHost {
    static constexpr int ORBIT_FULL_CAP = TARGET_W / 2;
    static constexpr int CLOSURE_FULL_CAP = (TARGET_W + 1) / 2;
    static constexpr int CAP_STRIDE = CLOSURE_FULL_CAP + 1;

    std::uint32_t masks = 0;
    // [mask * CAP_STRIDE + cap], task unit = one warp-row task.
    std::vector<std::uint32_t> orbit_warp_tasks;
    // [(mask * CAP_STRIDE + cap) * LOW_LUT_K + pi], task unit = one warp.
    std::vector<std::uint32_t> closure_warp_tasks;

    std::uint32_t orbit(std::uint32_t mask, int cap) const {
        cap = std::max(1, std::min(cap, ORBIT_FULL_CAP));
        return orbit_warp_tasks[
            std::size_t(mask) * CAP_STRIDE + std::size_t(cap)];
    }
    std::uint32_t closure(std::uint32_t mask, int cap, int pi) const {
        cap = std::max(1, std::min(cap, CLOSURE_FULL_CAP));
        return closure_warp_tasks[
            (std::size_t(mask) * CAP_STRIDE + std::size_t(cap)) * LOW_LUT_K
            + std::size_t(pi)];
    }
};

static MaskShardLowMaskBatchTablesHost maskshard_build_low_maskbatch_tables() {
    constexpr int H = HIGH_LUT_K;
    constexpr int ORBIT_FULL_CAP = TARGET_W / 2;
    constexpr int CLOSURE_FULL_CAP = (TARGET_W + 1) / 2;
    constexpr int CAP_STRIDE = CLOSURE_FULL_CAP + 1;
    constexpr std::uint32_t NMASKS = 1u << H;

    if (!G_MS_LOW_CLOSURE_PREFIX_HOST.built) {
        std::cerr << "LOW mask-batch task tables require LOW closure host capture\n";
        std::exit(345);
    }

    auto& orbit_cache = maskshard_loworbit_rowdepth_compact_cache();
    orbit_cache.build();
    const auto& closure_host = G_MS_LOW_CLOSURE_PREFIX_HOST;

    MaskShardLowMaskBatchTablesHost out;
    out.masks = NMASKS;
    out.orbit_warp_tasks.assign(
        std::size_t(NMASKS) * CAP_STRIDE, std::uint32_t(0));
    out.closure_warp_tasks.assign(
        std::size_t(NMASKS) * CAP_STRIDE * LOW_LUT_K, std::uint32_t(0));

    for (std::uint32_t mask = 0; mask < NMASKS; ++mask) {
        const MaskShardLowGroupPackedConfig base =
            maskshard_build_low_group_packed_base_uncached(mask);

        for (int cap = 1; cap <= CLOSURE_FULL_CAP; ++cap) {
            // Orbit saturates one cap earlier for even TARGET_W.
            const int orbit_cap = std::min(cap, ORBIT_FULL_CAP);
            std::array<Code, H + 3> state_prefix{};
            std::array<std::uint32_t, H + 2> low_count{};
            orbit_cache.make_job_plan(mask, orbit_cap, state_prefix, low_count);

            std::uint64_t orbit_warps = 0;
            for (int h = 0; h <= H + 1; ++h) {
                const Code states = state_prefix[std::size_t(h + 1)]
                                  - state_prefix[std::size_t(h)];
                const std::uint32_t lc = low_count[std::size_t(h)];
                const Code hc = lc ? states / Code(lc) : 0;
                if (lc && hc * Code(lc) != states) {
                    std::cerr << "LOW mask-batch orbit non-Cartesian mask="
                              << mask << " cap=" << cap << " h=" << h << '\n';
                    std::exit(346);
                }
                orbit_warps += std::uint64_t(hc) * ((lc + 31u) >> 5);
            }
            if (orbit_warps > 0xffffffffULL) {
                std::cerr << "LOW mask-batch orbit task overflow mask="
                          << mask << " cap=" << cap << '\n';
                std::exit(347);
            }
            out.orbit_warp_tasks[
                std::size_t(mask) * CAP_STRIDE + std::size_t(cap)] =
                std::uint32_t(orbit_warps);

            for (int pi = 0; pi < LOW_LUT_K; ++pi) {
                std::uint64_t closure_warps = 0;
                for (int b = 0; b < base.main_nblocks; ++b) {
                    const FBlock x = base.main_blocks[b];
                    const std::uint32_t a = closure_host.block_off[
                        std::size_t(pi) * 65 + std::size_t(b)];
                    const std::uint32_t z = closure_host.block_off[
                        std::size_t(pi) * 65 + std::size_t(b + 1)];
                    const std::uint32_t selected = cap >= CLOSURE_FULL_CAP
                        ? z - a
                        : closure_host.compact_active_count[
                            (std::size_t(pi) * 65 + std::size_t(b)) * CAP_STRIDE
                            + std::size_t(cap)];
                    const std::uint32_t rows = cap >= CLOSURE_FULL_CAP
                        ? (x.stride
                            ? std::uint32_t((x.end - x.off) / x.stride)
                            : 0u)
                        : std::uint32_t(closure_host.high_active_count[
                            (std::size_t(mask) * (H + 2) + x.he) * CAP_STRIDE
                            + std::size_t(cap)]);
                    closure_warps += std::uint64_t(rows)
                                   * ((selected + 31u) >> 5);
                }
                if (closure_warps > 0xffffffffULL) {
                    std::cerr << "LOW mask-batch closure task overflow mask="
                              << mask << " cap=" << cap << " pi=" << pi << '\n';
                    std::exit(348);
                }
                out.closure_warp_tasks[
                    (std::size_t(mask) * CAP_STRIDE + std::size_t(cap))
                        * LOW_LUT_K + std::size_t(pi)] =
                    std::uint32_t(closure_warps);
            }
        }
    }

    std::cerr << "LOW mask-batch exact task tables masks=" << out.masks
              << " host_mib="
              << double(out.orbit_warp_tasks.size() * sizeof(std::uint32_t)
                      + out.closure_warp_tasks.size() * sizeof(std::uint32_t))
                    / double(1ULL << 20) << '\n';
    return out;
}
