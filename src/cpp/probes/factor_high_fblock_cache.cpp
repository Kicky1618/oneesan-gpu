#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>

using U64 = std::uint64_t;

struct ModeledFBlock {
    U64 off, end;
    std::uint32_t stride;
    std::uint8_t he, hs, c, pad;
};
static_assert(sizeof(ModeledFBlock) == 24, "FBlock payload model drift");

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const int low = argc > 2 ? std::atoi(argv[2]) : 14;
    const int high = W - 1 - low;
    if (W < 4 || low < 1 || high < 1 || low >= 31) return 1;

    const U64 masks = U64(1) << low;
    const U64 rows = U64(W);
    const U64 jobs = masks * rows;
    const U64 main_blocks = U64(3) * U64(high + 2);
    const U64 blocked_blocks = U64(high + 2);
    const U64 blocks_per_mask = main_blocks + blocked_blocks;
    const U64 cache_entries = masks * blocks_per_mask;
    const U64 cache_payload_bytes = cache_entries * sizeof(ModeledFBlock);
    const U64 baseline_runtime_vector_builds = jobs * 2;
    const U64 cached_runtime_vector_builds = 0;
    const U64 setup_vector_builds = masks * 2;

    if (W == 28 && low == 14) {
        if (masks != 16384ULL
            || jobs != 458752ULL
            || main_blocks != 45ULL
            || blocked_blocks != 15ULL
            || cache_entries != 983040ULL
            || cache_payload_bytes != 23592960ULL
            || baseline_runtime_vector_builds != 917504ULL
            || setup_vector_builds != 32768ULL) {
            std::cerr << "n=27 HIGH FBlock cache regression mismatch\n";
            return 2;
        }
    }

    std::cout << std::fixed << std::setprecision(9)
              << "high-fblock-cache W=" << W
              << " low=" << low << " high=" << high << '\n'
              << "masks=" << masks << " rows=" << rows
              << " high_jobs_per_residue=" << jobs << '\n'
              << "main_blocks_per_mask=" << main_blocks
              << " blocked_blocks_per_mask=" << blocked_blocks
              << " cache_entries=" << cache_entries << '\n'
              << "cache_payload_bytes=" << cache_payload_bytes
              << " cache_payload_mib="
              << double(cache_payload_bytes) / double(1ULL << 20) << '\n'
              << "baseline_runtime_vector_builds_per_residue="
              << baseline_runtime_vector_builds
              << " cached_runtime_vector_builds_per_residue="
              << cached_runtime_vector_builds
              << " setup_vector_builds=" << setup_vector_builds << '\n';
    return 0;
}
