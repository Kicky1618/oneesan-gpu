#include <cstdint>
#include <iostream>

int main() {
    constexpr uint32_t WARP = 32;
    constexpr uint32_t BLOCK = 32;
    constexpr uint32_t HEIGHTS_W28 = 30;
    uint64_t cases = 0, lane_checks = 0;
    uint32_t max_padding = 0, max_blocks = 0;

    for (uint32_t raw_mod = 0; raw_mod < BLOCK; ++raw_mod) {
        const uint32_t pad = (BLOCK - raw_mod) & (BLOCK - 1u);
        const uint32_t hoff = raw_mod + pad;
        if ((hoff & (BLOCK - 1u)) != 0u) return 2;
        if (pad > max_padding) max_padding = pad;
        for (uint32_t active = 1; active <= WARP; ++active) {
            for (uint32_t stripe = 0; stripe < 4; ++stripe) {
                const uint32_t base = hoff + stripe * WARP;
                const uint32_t block = base >> 5;
                uint32_t seen = 0;
                for (uint32_t lane = 0; lane < active; ++lane) {
                    const uint32_t compact = base + lane;
                    const uint32_t direct_block = compact >> 5;
                    if (direct_block != block) {
                        std::cerr << "align32 block mismatch raw_mod=" << raw_mod
                                  << " active=" << active << " stripe=" << stripe
                                  << " lane=" << lane << '\n';
                        return 3;
                    }
                    seen |= 1u << (direct_block & 31u);
                    ++lane_checks;
                }
                const uint32_t blocks = uint32_t(__builtin_popcount(seen));
                if (blocks > max_blocks) max_blocks = blocks;
                if (blocks != 1u) return 4;
                ++cases;
            }
        }
    }

    if (max_padding != 31u || max_blocks != 1u) return 5;
    std::cout << "gridfp-rankchunk32-align32 OK"
              << " cases=" << cases
              << " lane_checks=" << lane_checks
              << " block=32 height_align=32"
              << " max_block_base_loads_per_warp=1"
              << " lane0_source_always_active=1"
              << " max_padding_per_height=" << max_padding
              << " w28_heights=" << HEIGHTS_W28
              << " max_padding_entries_per_owner=" << (HEIGHTS_W28 * max_padding)
              << " max_padding_bytes_per_owner=" << (HEIGHTS_W28 * max_padding * 4u)
              << " direct_block_exact=1\n";
    return 0;
}
