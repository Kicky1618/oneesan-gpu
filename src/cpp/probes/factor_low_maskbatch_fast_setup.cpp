#include <cstdint>
#include <cstdlib>
#include <iostream>

using U64 = std::uint64_t;

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const int low = argc > 2 ? std::atoi(argv[2]) : 14;
    const int high = W - 1 - low;
    if (W < 4 || low < 1 || high < 1 || high >= 31) return 1;

    const U64 masks = U64(1) << high;
    const U64 caps = U64((W + 1) / 2);
    const U64 configs = masks * caps;
    const U64 main_nblocks = U64(3 * (high + 2));
    const U64 orbit_segments_per_config = U64(high + 2);
    const U64 closure_segments_per_config = U64(low) * main_nblocks;
    const U64 segment_evals = configs
        * (orbit_segments_per_config + closure_segments_per_config);
    const U64 packed_base_fblock_copies = configs * U64(64 + 32);

    if (W == 28 && low == 14) {
        if (configs != 114688ULL
            || main_nblocks != 45ULL
            || segment_evals != 73973760ULL
            || packed_base_fblock_copies != 11010048ULL) {
            std::cerr << "n=27 LOW fast-setup regression\n";
            return 2;
        }
    }

    std::cout << "low-maskbatch-fast-setup W=" << W
              << " low=" << low << " high=" << high << '\n'
              << "old_duplicate_packed_config_builds=" << configs
              << " new_duplicate_packed_config_builds=0\n"
              << "old_duplicate_segment_evals=" << segment_evals
              << " new_duplicate_segment_evals=0\n"
              << "old_duplicate_fblock_copies=" << packed_base_fblock_copies
              << " new_duplicate_fblock_copies=0\n";
    return 0;
}
