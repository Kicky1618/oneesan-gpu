#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>

using U64 = std::uint64_t;

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const int low = argc > 2 ? std::atoi(argv[2]) : 14;
    const int ngpu = argc > 3 ? std::atoi(argv[3]) : 8;
    const double launch_us = argc > 4 ? std::atof(argv[4]) : 3.0;
    const int high = W - 1 - low;
    if (W < 4 || low < 1 || high < 1 || high >= 31 || ngpu < 1) return 1;

    const U64 masks = U64(1) << high;
    const U64 group_calls = masks * U64(W);
    const U64 kernels_per_group = U64(2 * low);
    const U64 current_launches = group_calls * kernels_per_group;
    const U64 batched_launches = U64(ngpu) * U64(W) * kernels_per_group;
    const U64 current_config_calls = group_calls;
    const U64 batched_row_updates = U64(ngpu) * U64(W);

    // v0.42 n=27 packed-config field sizes excluding ABI tail padding.
    const U64 static_bytes = U64(96 * 24 + 16);
    const U64 dynamic_bytes = U64((high + 3) * 4)
        + U64(high + 2) * 4 * 2
        + U64(low) * 65 * 4 * 3
        + U64(high + 2) * 4;
    const U64 masks_per_gpu = (masks + U64(ngpu) - 1) / U64(ngpu);
    const U64 all_cap_bytes_per_gpu =
        masks_per_gpu * (static_bytes + U64(low) * dynamic_bytes);

    if (W == 28 && low == 14 && ngpu == 8) {
        if (current_launches != 6422528ULL
            || batched_launches != 6272ULL
            || current_config_calls != 229376ULL
            || batched_row_updates != 224ULL
            || dynamic_bytes != 11164ULL
            || all_cap_bytes_per_gpu != 162422784ULL) {
            std::cerr << "n=27 LOW mask-batch regression\n";
            return 2;
        }
    }

    std::cout << std::fixed << std::setprecision(6)
              << "low-maskbatch-launch W=" << W << " low=" << low
              << " high=" << high << " gpus=" << ngpu << '\n'
              << "current_kernel_launches=" << current_launches
              << " batched_kernel_launches=" << batched_launches
              << " launch_reduction="
              << (1.0 - double(batched_launches) / double(current_launches)) << '\n'
              << "current_group_config_calls=" << current_config_calls
              << " batched_row_config_updates=" << batched_row_updates << '\n'
              << "assumed_launch_us=" << launch_us
              << " current_launch_overhead_s="
              << double(current_launches) * launch_us / 1.0e6
              << " batched_launch_overhead_s="
              << double(batched_launches) * launch_us / 1.0e6 << '\n'
              << "all_cap_config_bytes_per_gpu=" << all_cap_bytes_per_gpu
              << " all_cap_config_mib_per_gpu="
              << double(all_cap_bytes_per_gpu) / double(1ULL << 20) << '\n'
              << "replica_descriptor_bytes_at_16x_per_mask_total="
              << masks * 16ULL * 8ULL << '\n';
    return 0;
}
