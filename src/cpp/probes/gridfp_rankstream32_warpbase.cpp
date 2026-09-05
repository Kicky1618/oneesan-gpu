#include <cstdint>
#include <cstdlib>
#include <iostream>

int main() {
    constexpr uint32_t WARP = 32;
    uint64_t cases = 0, lanes = 0;
    uint32_t max_blocks = 0;

    // height_off modulo 32 is the only alignment freedom that matters.
    // rank stripes always begin at a multiple of 32 and active lanes on the
    // final stripe are the contiguous prefix [0, active_lanes).
    for (uint32_t hoff_mod = 0; hoff_mod < WARP; ++hoff_mod) {
        for (uint32_t active_lanes = 1; active_lanes <= WARP; ++active_lanes) {
            const uint32_t first_compact = hoff_mod;
            const uint32_t first_block = first_compact >> 5;
            const uint32_t first_off = first_compact & 31u;
            const uint32_t split_lane = WARP - first_off;
            uint32_t used_mask = 0;
            for (uint32_t lane = 0; lane < active_lanes; ++lane) {
                const uint32_t compact = hoff_mod + lane;
                const uint32_t direct_block = compact >> 5;
                uint32_t shared_block = first_block;
                if (split_lane < WARP && lane >= split_lane)
                    shared_block = first_block + 1u;
                if (direct_block != shared_block) {
                    std::cerr << "rankstream32 warp-base mismatch hoff_mod=" << hoff_mod
                              << " active=" << active_lanes << " lane=" << lane
                              << " direct=" << direct_block << " shared=" << shared_block << '\n';
                    return 2;
                }
                used_mask |= 1u << direct_block;
                ++lanes;
            }
            uint32_t blocks = uint32_t(__builtin_popcount(used_mask));
            if (blocks > 2u) return 3;
            max_blocks = blocks > max_blocks ? blocks : max_blocks;

            // lane 0 is always active. If the second block is used, the lane
            // at which it starts is also necessarily active and can broadcast.
            if (active_lanes == 0) return 4;
            if (blocks == 2u) {
                if (!(split_lane < WARP && split_lane < active_lanes)) {
                    std::cerr << "rankstream32 second-block source inactive hoff_mod=" << hoff_mod
                              << " active=" << active_lanes << " split=" << split_lane << '\n';
                    return 5;
                }
            }
            ++cases;
        }
    }

    if (cases != 32u * 32u || max_blocks != 2u) return 6;
    std::cout << "gridfp-rankstream32-warpbase OK"
              << " cases=" << cases
              << " active_lane_checks=" << lanes
              << " alignments=32 partial_widths=32"
              << " max_block_base_loads_per_warp=2"
              << " lane0_source_always_active=1"
              << " second_source_active_if_needed=1"
              << " direct_block_exact=1\n";
    return 0;
}
