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

    const U64 warp_prefix = U64(high + 3) * 4ULL;
    const U64 low_count_chunks = U64(high + 2) * 2ULL * 4ULL;
    const U64 closure_prefix = U64(low) * 65ULL * 4ULL;
    const U64 closure_begin_selected = U64(low) * 65ULL * 2ULL * 4ULL;
    const U64 high_mask_off = U64(high + 2) * 4ULL;
    const U64 old_bytes = warp_prefix + low_count_chunks + closure_prefix
                        + closure_begin_selected + high_mask_off;
    const U64 compact_bytes = old_bytes - closure_begin_selected;
    const U64 old_per_gpu = masks_per_gpu * caps * old_bytes;
    const U64 compact_per_gpu = masks_per_gpu * caps * compact_bytes;
    const U64 saved_per_gpu = old_per_gpu - compact_per_gpu;

    if (W == 28 && low == 14 && ngpu == 8) {
        if (old_bytes != 11164ULL
            || compact_bytes != 3884ULL
            || old_per_gpu != 160047104ULL
            || compact_per_gpu != 55681024ULL
            || saved_per_gpu != 104366080ULL) {
            std::cerr << "n=27 LOW compact-dynamic regression\n";
            return 2;
        }
    }

    std::cout << std::fixed << std::setprecision(6)
              << "low-maskbatch-compact-dynamic W=" << W
              << " low=" << low << " high=" << high
              << " gpus=" << ngpu << '\n'
              << "old_dynamic_bytes_per_mask_cap=" << old_bytes
              << " compact_dynamic_bytes_per_mask_cap=" << compact_bytes
              << " reduction="
              << (1.0 - double(compact_bytes) / double(old_bytes)) << '\n'
              << "old_dynamic_mib_per_gpu="
              << double(old_per_gpu) / double(1ULL << 20)
              << " compact_dynamic_mib_per_gpu="
              << double(compact_per_gpu) / double(1ULL << 20)
              << " saved_mib_per_gpu="
              << double(saved_per_gpu) / double(1ULL << 20) << '\n';
    return 0;
}
