#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "maskshard_low_maskbatch_plan.hpp"
#include "maskshard_low_maskbatch_tables.hpp"

struct MaskShardLowMaskBatchRowPlan {
    // Orbit descriptors are shared by every LOW p at one row because the exact
    // row-depth orbit task space is p-independent.
    std::array<std::vector<MaskShardLowBatchDesc>, 8> orbit;
    // Closure task count depends on p (pi), so keep one descriptor list/stage.
    std::array<std::array<std::vector<MaskShardLowBatchDesc>, LOW_LUT_K>, 8>
        closure;
};

static MaskShardLowMaskBatchRowPlan maskshard_build_low_maskbatch_row_plan(
    const MaskShardLayout& shard,
    const MaskShardLowMaskBatchTablesHost& tables,
    int zero_based_row,
    std::uint64_t target_tasks_per_cta = 16384,
    int max_replicas = 16
) {
    MaskShardLowMaskBatchRowPlan out;
    const int orbit_cap = std::min(zero_based_row + 1, TARGET_W / 2);
    const int closure_cap = std::min(zero_based_row + 1, (TARGET_W + 1) / 2);

    std::vector<std::uint64_t> tasks(std::size_t(shard.masks));
    for (std::uint32_t mask = 0; mask < shard.masks; ++mask)
        tasks[mask] = tables.orbit(mask, orbit_cap);
    for (int d = 0; d < shard.ngpu; ++d)
        out.orbit[d] = maskshard_build_low_batch_plan(
            shard.owner, d, tasks, target_tasks_per_cta, max_replicas);

    for (int pi = 0; pi < LOW_LUT_K; ++pi) {
        for (std::uint32_t mask = 0; mask < shard.masks; ++mask)
            tasks[mask] = tables.closure(mask, closure_cap, pi);
        for (int d = 0; d < shard.ngpu; ++d)
            out.closure[d][pi] = maskshard_build_low_batch_plan(
                shard.owner, d, tasks, target_tasks_per_cta, max_replicas);
    }
    return out;
}
