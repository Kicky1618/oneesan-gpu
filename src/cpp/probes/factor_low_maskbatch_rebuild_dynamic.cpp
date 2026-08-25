#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>

using U64 = std::uint64_t;

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const int low = argc > 2 ? std::atoi(argv[2]) : 14;
    const int ngpu = argc > 3 ? std::atoi(argv[3]) : 8;
    const int high = W - 1 - low;
    if (W < 4 || low < 1 || high < 1 || high >= 31 || ngpu < 1) return 1;

    const U64 masks = U64(1) << high;
    const U64 masks_per_gpu = (masks + U64(ngpu) - 1) / U64(ngpu);
    const U64 caps = U64((W + 1) / 2);
    const U64 v47_dynamic_bytes =
        U64(high + 3) * 4ULL
        + U64(high + 2) * 2ULL * 4ULL
        + U64(low) * 65ULL * 4ULL
        + U64(high + 2) * 4ULL;
    const U64 v47_per_gpu = masks_per_gpu * caps * v47_dynamic_bytes;
    const U64 low_count_constant_bytes =
        U64((W + 1) / 2 + 1) * U64(high + 2) * 4ULL;
    const U64 static_base_bytes_per_gpu = masks_per_gpu * 2320ULL;

    if (W == 28 && low == 14 && ngpu == 8) {
        if (v47_dynamic_bytes != 3884ULL
            || v47_per_gpu != 55681024ULL
            || low_count_constant_bytes != 900ULL
            || static_base_bytes_per_gpu != 2375680ULL) {
            std::cerr << "n=27 LOW rebuild-dynamic regression\n";
            return 2;
        }
    }

    std::cout << std::fixed << std::setprecision(6)
              << "low-maskbatch-rebuild-dynamic W=" << W
              << " low=" << low << " high=" << high
              << " gpus=" << ngpu << '\n'
              << "v47_dynamic_bytes_per_mask_cap=" << v47_dynamic_bytes
              << " v48_dynamic_bytes_per_mask_cap=0\n"
              << "v47_dynamic_mib_per_gpu="
              << double(v47_per_gpu) / double(1ULL << 20)
              << " v48_dynamic_mib_per_gpu=0"
              << " saved_mib_per_gpu="
              << double(v47_per_gpu) / double(1ULL << 20) << '\n'
              << "v48_low_count_constant_bytes=" << low_count_constant_bytes
              << " static_base_mib_per_gpu="
              << double(static_base_bytes_per_gpu) / double(1ULL << 20) << '\n';
    return 0;
}
