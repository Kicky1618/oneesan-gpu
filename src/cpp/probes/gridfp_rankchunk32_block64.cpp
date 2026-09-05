#include <cstdint>
#include <iostream>

int main() {
    constexpr uint32_t WARP = 32;
    constexpr uint32_t BLOCK = 64;
    constexpr uint32_t MAX_L = 7;
    constexpr uint32_t PREFIX_LIMIT = 512;
    constexpr uint32_t MAX_PREFIX = (BLOCK - 1u) * MAX_L;
    static_assert(MAX_PREFIX == 441u && MAX_PREFIX < PREFIX_LIMIT);

    uint64_t cases = 0, lane_checks = 0;
    uint32_t max_blocks = 0, crossing_alignments = 0;
    for (uint32_t off = 0; off < BLOCK; ++off) {
        bool alignment_crosses = false;
        for (uint32_t active = 1; active <= WARP; ++active) {
            const uint32_t first_block = off >> 6;
            const uint32_t split = BLOCK - (off & (BLOCK - 1u));
            uint32_t used0 = 0, used1 = 0;
            for (uint32_t lane = 0; lane < active; ++lane) {
                const uint32_t direct = (off + lane) >> 6;
                const uint32_t shared = first_block + ((split < WARP && lane >= split) ? 1u : 0u);
                if (direct != shared) return 2;
                if (direct == first_block) used0 = 1; else used1 = 1;
                ++lane_checks;
            }
            const uint32_t blocks = used0 + used1;
            if (blocks > max_blocks) max_blocks = blocks;
            if (blocks == 2u) alignment_crosses = true;
            if (blocks > 2u) return 3;
            ++cases;
        }
        if (alignment_crosses) ++crossing_alignments;
    }
    if (max_blocks != 2u || crossing_alignments != 31u) return 4;
    std::cout << "gridfp-rankchunk32-block64 OK"
              << " cases=" << cases
              << " lane_checks=" << lane_checks
              << " block=64 prefix_bits=9"
              << " max_l_per_legal_code=7 max_prefix=" << MAX_PREFIX
              << " max_block_base_loads_per_warp=2"
              << " crossing_alignments_full_warp=" << crossing_alignments
              << " noncrossing_alignments_full_warp=" << (BLOCK - crossing_alignments)
              << " block_table_reduction_vs32=2x"
              << " direct_block_exact=1\n";
    return 0;
}
