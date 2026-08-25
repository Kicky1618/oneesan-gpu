#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

#include "maskshard_low_maskbatch_range.hpp"
#include "maskshard_low_maskbatch_rowplan.hpp"

struct MaskShardLowMaskBatchRangeRowPlan {
    std::array<std::vector<MaskShardLowBatchRangeHostDesc>, 8> orbit;
    std::array<
        std::array<std::vector<MaskShardLowBatchRangeHostDesc>, LOW_LUT_K>, 8>
        closure;
};

static MaskShardLowMaskBatchRangeRowPlan
maskshard_build_low_maskbatch_range_row_plan(
    const MaskShardLayout& shard,
    const MaskShardLowMaskBatchTablesHost& tables,
    int zero_based_row,
    std::uint64_t target_tasks_per_cta,
    int max_replicas
) {
    MaskShardLowMaskBatchRangeRowPlan out;
    const int orbit_cap = std::min(zero_based_row + 1, TARGET_W / 2);
    const int closure_cap = std::min(zero_based_row + 1, (TARGET_W + 1) / 2);

    std::vector<std::uint64_t> tasks(std::size_t(shard.masks));
    for (std::uint32_t mask = 0; mask < shard.masks; ++mask)
        tasks[mask] = tables.orbit(mask, orbit_cap);
    for (int d = 0; d < shard.ngpu; ++d)
        out.orbit[d] = maskshard_build_low_batch_range_plan(
            shard.owner, d, tasks, target_tasks_per_cta, max_replicas);

    for (int pi = 0; pi < LOW_LUT_K; ++pi) {
        for (std::uint32_t mask = 0; mask < shard.masks; ++mask)
            tasks[mask] = tables.closure(mask, closure_cap, pi);
        for (int d = 0; d < shard.ngpu; ++d)
            out.closure[d][pi] = maskshard_build_low_batch_range_plan(
                shard.owner, d, tasks, target_tasks_per_cta, max_replicas);
    }
    return out;
}
