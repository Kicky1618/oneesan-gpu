#include <cstdint>
#include <iostream>

int main() {
    constexpr uint32_t WARP = 32;
    constexpr uint32_t BLOCK = 16;
    constexpr uint32_t ALIGN = 32;
    uint64_t cases = 0, lanes = 0;
    uint32_t max_blocks = 0, max_padding = 0;

    // Before a height is emitted its metadata cursor can have any alignment.
    // Padding moves the height start to the next 32-entry boundary.  Rank
    // stripes then begin at another multiple of 32, so lanes 0..15 and 16..31
    // map exactly to the two 16-entry metadata blocks.
    for (uint32_t prior_mod = 0; prior_mod < ALIGN; ++prior_mod) {
        const uint32_t padding = (ALIGN - prior_mod) & (ALIGN - 1u);
        if (padding > max_padding) max_padding = padding;
        const uint32_t hoff = prior_mod + padding;
        if ((hoff & (ALIGN - 1u)) != 0u) return 2;

        for (uint32_t stripe = 0; stripe < 4u * WARP; stripe += WARP) {
            for (uint32_t active_lanes = 1; active_lanes <= WARP; ++active_lanes) {
                const uint32_t first_compact = hoff + stripe;
                const uint32_t first_block = first_compact >> 4;
                uint32_t blocks = 1;

                for (uint32_t lane = 0; lane < active_lanes; ++lane) {
                    const uint32_t compact = hoff + stripe + lane;
                    const uint32_t direct_block = compact >> 4;
                    const uint32_t shared_block = first_block + (lane >= BLOCK ? 1u : 0u);
                    if (direct_block != shared_block) {
                        std::cerr << "rankchunk32 aligned warp-base mismatch prior_mod=" << prior_mod
                                  << " padding=" << padding << " stripe=" << stripe
                                  << " active=" << active_lanes << " lane=" << lane
                                  << " direct=" << direct_block << " shared=" << shared_block << '\n';
                        return 3;
                    }
                    if (lane >= BLOCK) blocks = 2;
                    ++lanes;
                }

                if (blocks > max_blocks) max_blocks = blocks;
                if (blocks == 2u && active_lanes <= BLOCK) return 4;
                ++cases;
            }
        }
    }

    if (cases != 32u * 4u * 32u || max_blocks != 2u || max_padding != 31u) return 5;
    std::cout << "gridfp-rankchunk32-warpbase OK"
              << " cases=" << cases
              << " active_lane_checks=" << lanes
              << " prealignments=32 stripes=4 partial_widths=32"
              << " height_alignment=32"
              << " max_padding_entries_per_height=31"
              << " max_block_base_loads_per_warp=2"
              << " lane0_source_always_active=1"
              << " lane16_source_active_if_needed=1"
              << " direct_block_exact=1\n";
    return 0;
}
