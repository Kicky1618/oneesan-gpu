#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "maskshard_low_maskbatch_rowplan.hpp"

struct MaskShardLowBatchRangeHostDesc {
    std::uint16_t mask = 0;
    std::uint16_t replicas = 0;
};
static_assert(sizeof(MaskShardLowBatchRangeHostDesc) == 4,
              "LOW mask-batch range host descriptor ABI changed");

static std::vector<MaskShardLowBatchRangeHostDesc>
maskshard_build_low_batch_range_plan(
    const std::vector<std::uint8_t>& owner,
    int device,
    const std::vector<std::uint64_t>& task_count,
    std::uint64_t target_tasks_per_cta,
    int max_replicas
) {
    if (owner.size() != task_count.size() || device < 0
        || target_tasks_per_cta == 0 || max_replicas < 1
        || max_replicas > 65535 || owner.size() > 0xffffu) {
        std::cerr << "invalid LOW mask-batch range plan arguments\n";
        std::exit(354);
    }

    std::vector<MaskShardLowBatchRangeHostDesc> out;
    out.reserve(owner.size());
    for (std::uint32_t mask = 0; mask < owner.size(); ++mask) {
        if (int(owner[mask]) != device) continue;
        const std::uint64_t tasks = task_count[mask];
        if (!tasks) continue;
        const std::uint64_t wanted =
            (tasks + target_tasks_per_cta - 1) / target_tasks_per_cta;
        const std::uint64_t replicas = std::min<std::uint64_t>(
            std::uint64_t(max_replicas), std::max<std::uint64_t>(1, wanted));
        out.push_back({std::uint16_t(mask), std::uint16_t(replicas)});
    }
    return out;
}

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
