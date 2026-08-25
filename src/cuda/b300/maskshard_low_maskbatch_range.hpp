#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

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
