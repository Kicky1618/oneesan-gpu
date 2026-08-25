#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>

using U64 = std::uint64_t;

static U64 align8(U64 x) { return (x + 7ULL) & ~7ULL; }

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const int low = argc > 2 ? std::atoi(argv[2]) : 14;
    const U64 target_warp_tasks = argc > 3
        ? std::strtoull(argv[3], nullptr, 10) : 16384ULL;
    const int high = W - 1 - low;
    if (W < 4 || low < 1 || high < 1 || target_warp_tasks == 0) return 1;

    // FBlock is pinned to 24 bytes by the factorized storage ABI.  v0.46 stages
    // all static MAIN/BLOCKED blocks plus only the current stage's hot uint32
    // metadata.  Struct size is rounded to FBlock's 8-byte alignment.
    const U64 fblock_bytes = 24;
    const U64 static_fblocks = U64(64 + 32) * fblock_bytes;
    const U64 orbit_u32 = U64(high + 3) + 3ULL * U64(high + 2);
    const U64 closure_u32 = 65ULL * 3ULL + U64(high + 2);
    const U64 orbit_shared = align8(static_fblocks + orbit_u32 * 4ULL);
    const U64 closure_shared = align8(static_fblocks + closure_u32 * 4ULL);

    if (W == 28 && low == 14 && target_warp_tasks == 16384ULL) {
        if (orbit_shared != 2552ULL || closure_shared != 3144ULL) {
            std::cerr << "n=27 LOW CTA-cache footprint regression\n";
            return 2;
        }
    }

    std::cout << std::fixed << std::setprecision(9)
              << "low-maskbatch-ctacache W=" << W
              << " low=" << low << " high=" << high
              << " target_warp_tasks_per_cta=" << target_warp_tasks << '\n'
              << "orbit_shared_bytes=" << orbit_shared
              << " closure_shared_bytes=" << closure_shared << '\n'
              << "orbit_stage_bytes_per_target_warp_task="
              << double(orbit_shared) / double(target_warp_tasks) << '\n'
              << "closure_stage_bytes_per_target_warp_task="
              << double(closure_shared) / double(target_warp_tasks) << '\n'
              << "cta_barriers_per_kernel_cta=1\n";
    return 0;
}
