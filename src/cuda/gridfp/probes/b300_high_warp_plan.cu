#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

#define RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../oneesan_cuda_gridfp_ramstream32_factorized_bidesc_compact.cu"
#undef RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../ramstream32_b300_high_warp.cuh"

int main() {
    build_full_dp();
    G_FACTOR = build_factor_tables();

    uint32_t masks = 1u << LOW_LUT_K;
    uint64_t main_rows_total = 0, block_rows_total = 0;
    uint64_t main_blocks_total = 0, block_blocks_total = 0;
    uint32_t main_rows_max = 0, block_rows_max = 0;
    uint32_t main_blocks_max = 0, block_blocks_max = 0;
    uint32_t stride_max = 0;
    uint64_t state_total = 0;

    for (uint32_t mask = 0; mask < masks; ++mask) {
        auto mb = make_factor_main_blocks(true, mask);
        auto db = make_factor_block_blocks(true, mask);
        uint64_t mr = 0, br = 0;
        for (const FBlock& b : mb) {
            if (!b.stride) continue;
            mr += (b.end - b.off) / b.stride;
            stride_max = std::max(stride_max, b.stride);
            state_total += b.end - b.off;
        }
        for (const FBlock& b : db) {
            if (!b.stride) continue;
            br += (b.end - b.off) / b.stride;
            stride_max = std::max(stride_max, b.stride);
        }
        if (mr > 0xffffffffULL || br > 0xffffffffULL) return 350;
        uint32_t mru = uint32_t(mr), bru = uint32_t(br);
        uint32_t mbk = high_warp_blocks(mru);
        uint32_t bbk = high_warp_blocks(bru);
        main_rows_total += mr;
        block_rows_total += br;
        main_blocks_total += mbk;
        block_blocks_total += bbk;
        main_rows_max = std::max(main_rows_max, mru);
        block_rows_max = std::max(block_rows_max, bru);
        main_blocks_max = std::max(main_blocks_max, mbk);
        block_blocks_max = std::max(block_blocks_max, bbk);
    }

    double avg_width = main_rows_total ? double(state_total) / double(main_rows_total) : 0.0;
    std::cout << std::fixed << std::setprecision(3)
        << "b300-high-warp-plan W=" << TARGET_W
        << " masks=" << masks
        << " main_rows_total=" << main_rows_total
        << " main_warp_blocks_total=" << main_blocks_total
        << " row_to_block_reduction="
        << (main_blocks_total ? double(main_rows_total) / double(main_blocks_total) : 0.0)
        << " main_rows_max_group=" << main_rows_max
        << " main_warp_blocks_max_group=" << main_blocks_max
        << " block_rows_total=" << block_rows_total
        << " block_warp_blocks_total=" << block_blocks_total
        << " block_rows_max_group=" << block_rows_max
        << " block_warp_blocks_max_group=" << block_blocks_max
        << " main_avg_low_width=" << avg_width
        << " low_width_max=" << stride_max
        << '\n';
    return 0;
}
