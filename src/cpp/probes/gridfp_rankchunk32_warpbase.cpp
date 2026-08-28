#include <cstdint>
#include <iostream>

int main() {
    constexpr uint32_t WARP = 32;
    constexpr uint32_t BLOCK = 16;
    uint64_t cases = 0, lanes = 0;
    uint32_t max_blocks = 0;

    // A warp stripe starts at a rank multiple of 32.  Only the compact height
    // offset modulo the 16-code metadata block changes the block alignment.
    // Final partial stripes keep active lanes as the contiguous prefix
    // [0,active_lanes), exactly as the warp-striped HIGH kernels do.
    for (uint32_t hoff_mod = 0; hoff_mod < BLOCK; ++hoff_mod) {
        for (uint32_t active_lanes = 1; active_lanes <= WARP; ++active_lanes) {
            const uint32_t first_compact = hoff_mod;
            const uint32_t first_block = first_compact >> 4;
            const uint32_t first_off = first_compact & 15u;
            const uint32_t split1 = BLOCK - first_off;
            const uint32_t split2 = split1 + BLOCK;
            uint32_t used_mask = 0;

            for (uint32_t lane = 0; lane < active_lanes; ++lane) {
                const uint32_t compact = hoff_mod + lane;
                const uint32_t direct_block = compact >> 4;
                uint32_t shared_block = first_block;
                if (lane >= split1) shared_block = first_block + 1u;
                if (split2 < WARP && lane >= split2) shared_block = first_block + 2u;
                if (direct_block != shared_block) {
                    std::cerr << "rankchunk32 warp-base mismatch hoff_mod=" << hoff_mod
                              << " active=" << active_lanes << " lane=" << lane
                              << " direct=" << direct_block << " shared=" << shared_block
                              << " split1=" << split1 << " split2=" << split2 << '\n';
                    return 2;
                }
                used_mask |= 1u << direct_block;
                ++lanes;
            }

            const uint32_t blocks = uint32_t(__builtin_popcount(used_mask));
            if (blocks > 3u) return 3;
            if (blocks > max_blocks) max_blocks = blocks;

            // lane 0 is always active.  A later block is used iff its first lane
            // is active, so that lane can safely issue the load and broadcast.
            if (blocks >= 2u && !(split1 < active_lanes)) {
                std::cerr << "rankchunk32 second-block source inactive hoff_mod=" << hoff_mod
                          << " active=" << active_lanes << " split1=" << split1 << '\n';
                return 4;
            }
            if (blocks == 3u && !(split2 < WARP && split2 < active_lanes)) {
                std::cerr << "rankchunk32 third-block source inactive hoff_mod=" << hoff_mod
                          << " active=" << active_lanes << " split2=" << split2 << '\n';
                return 5;
            }
            ++cases;
        }
    }

    if (cases != 16u * 32u || max_blocks != 3u) return 6;
    std::cout << "gridfp-rankchunk32-warpbase OK"
              << " cases=" << cases
              << " active_lane_checks=" << lanes
              << " alignments=16 partial_widths=32"
              << " max_block_base_loads_per_warp=3"
              << " lane0_source_always_active=1"
              << " second_source_active_if_needed=1"
              << " third_source_active_if_needed=1"
              << " direct_block_exact=1\n";
    return 0;
}
