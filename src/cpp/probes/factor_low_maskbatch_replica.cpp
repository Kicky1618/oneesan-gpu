#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "../../cuda/b300/maskshard_low_maskbatch_plan.hpp"

int main(int argc, char** argv) {
    const int masks = argc > 1 ? std::atoi(argv[1]) : 257;
    const int warps = argc > 2 ? std::atoi(argv[2]) : 8;
    const int max_replicas = argc > 3 ? std::atoi(argv[3]) : 16;
    const std::uint64_t target_per_cta = argc > 4
        ? std::strtoull(argv[4], nullptr, 10) : 4096ULL;
    if (masks < 1 || masks > 65535 || warps < 1 || max_replicas < 1
        || max_replicas > 65535 || target_per_cta < 1)
        return 1;

    std::vector<std::uint64_t> tasks(size_t(masks));
    std::vector<std::uint8_t> owner(size_t(masks), std::uint8_t(0));
    for (int m = 0; m < masks; ++m) {
        // Deterministic mix including zero, tiny, boundary, and large groups.
        tasks[size_t(m)] = (m % 17 == 0) ? 0ULL
            : ((std::uint64_t(m + 3) * 11939ULL) % 250000ULL)
                + std::uint64_t(m % (warps + 1));
    }

    const std::vector<MaskShardLowBatchDesc> descs =
        maskshard_build_low_batch_plan(
            owner, 0, tasks, target_per_cta, max_replicas);

    std::vector<std::vector<std::uint8_t>> seen(size_t(masks));
    for (int m = 0; m < masks; ++m)
        seen[size_t(m)].assign(size_t(tasks[size_t(m)]), std::uint8_t(0));

    bool saw_wide_replica = false;
    std::uint16_t max_replica_field = 0;
    std::uint16_t max_replicas_field = 0;
    for (const MaskShardLowBatchDesc& d : descs) {
        max_replica_field = std::max(max_replica_field, d.replica);
        max_replicas_field = std::max(max_replicas_field, d.replicas);
        saw_wide_replica = saw_wide_replica || d.replica > 255u || d.replicas > 255u;
        const std::uint64_t total = tasks[d.mask];
        for (int w = 0; w < warps; ++w) {
            std::uint64_t task = std::uint64_t(d.replica) * std::uint64_t(warps)
                               + std::uint64_t(w);
            const std::uint64_t step =
                std::uint64_t(d.replicas) * std::uint64_t(warps);
            for (; task < total; task += step) {
                std::uint8_t& x = seen[d.mask][size_t(task)];
                if (x != 0) {
                    std::cerr << "duplicate mask=" << d.mask
                              << " task=" << task << '\n';
                    return 2;
                }
                x = 1;
            }
        }
    }

    std::uint64_t total_tasks = 0;
    for (int m = 0; m < masks; ++m) {
        total_tasks += tasks[size_t(m)];
        for (std::uint64_t q = 0; q < tasks[size_t(m)]; ++q) {
            if (seen[size_t(m)][size_t(q)] != 1) {
                std::cerr << "missing mask=" << m << " task=" << q << '\n';
                return 3;
            }
        }
    }

    if (max_replicas > 255 && !saw_wide_replica) {
        std::cerr << "wide replica ABI was not exercised max_replicas="
                  << max_replicas << '\n';
        return 4;
    }

    std::cout << "low-maskbatch-replica masks=" << masks
              << " warps_per_cta=" << warps
              << " max_replicas=" << max_replicas
              << " descriptors=" << descs.size()
              << " descriptor_bytes="
              << descs.size() * sizeof(MaskShardLowBatchDesc)
              << " max_replica_field=" << max_replica_field
              << " max_replicas_field=" << max_replicas_field
              << " wide_replica_exercised=" << (saw_wide_replica ? 1 : 0)
              << " total_tasks=" << total_tasks
              << " exact_cover=1\n";
    return 0;
}
