#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

struct MaskShardLowBatchDesc {
    std::uint16_t mask = 0;
    std::uint16_t replica = 0;
    std::uint16_t replicas = 0;
};
static_assert(sizeof(MaskShardLowBatchDesc) == 6,
              "LOW mask-batch host descriptor ABI changed");

static std::vector<MaskShardLowBatchDesc> maskshard_build_low_batch_plan(
    const std::vector<std::uint8_t>& owner,
    int device,
    const std::vector<std::uint64_t>& task_count,
    std::uint64_t target_tasks_per_cta = 16384,
    int max_replicas = 16
) {
    if (owner.size() != task_count.size() || device < 0
        || target_tasks_per_cta == 0 || max_replicas < 1 || max_replicas > 65535) {
        std::cerr << "invalid LOW mask-batch plan arguments\n";
        std::exit(343);
    }
    if (owner.size() > 0xffffu) {
        std::cerr << "LOW mask-batch uint16 mask overflow masks="
                  << owner.size() << '\n';
        std::exit(344);
    }

    std::vector<MaskShardLowBatchDesc> out;
    out.reserve(owner.size());
    for (std::uint32_t mask = 0; mask < owner.size(); ++mask) {
        if (int(owner[mask]) != device) continue;
        const std::uint64_t tasks = task_count[mask];
        if (!tasks) continue;
        const std::uint64_t wanted =
            (tasks + target_tasks_per_cta - 1) / target_tasks_per_cta;
        const int replicas = int(std::min<std::uint64_t>(
            std::uint64_t(max_replicas), std::max<std::uint64_t>(1, wanted)));
        for (int q = 0; q < replicas; ++q) {
            out.push_back({
                std::uint16_t(mask),
                std::uint16_t(q),
                std::uint16_t(replicas),
            });
        }
    }
    return out;
}
