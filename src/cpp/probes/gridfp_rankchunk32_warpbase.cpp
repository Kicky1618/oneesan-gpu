#include <cstdint>
#include <iostream>

namespace {
constexpr uint32_t WARP = 32;
constexpr uint32_t BLOCK = 32;
constexpr uint32_t CHUNK_BITS = 23;
constexpr uint32_t PREFIX_BITS = 9;
constexpr uint32_t CHUNK_MASK = (1u << CHUNK_BITS) - 1u;
constexpr uint32_t K = 14;

constexpr uint32_t pow3(uint32_t n) {
    return n ? 3u * pow3(n - 1u) : 1u;
}
constexpr uint32_t pack_chunks(uint32_t key) {
    const uint32_t c0 = (key / pow3(9)) % pow3(5);
    const uint32_t c1 = (key / pow3(4)) % pow3(5);
    const uint32_t c2 = key % pow3(4);
    return c0 | (c1 << 8) | (c2 << 16);
}
constexpr uint32_t unpack_chunks(uint32_t packed) {
    return (packed & 0xffu) * pow3(9)
         + ((packed >> 8) & 0xffu) * pow3(4)
         + ((packed >> 16) & 0x7fu);
}
} // namespace

int main() {
    static_assert(pow3(4) == 81u);
    static_assert(pow3(5) == 243u);
    static_assert(31u * K < (1u << PREFIX_BITS));

    uint64_t pack_cases = 0;
    uint32_t max_packed = 0, max_tail = 0;
    for (uint32_t key = 0; key < pow3(K); ++key) {
        const uint32_t packed = pack_chunks(key);
        const uint32_t tail = (packed >> 16) & 0x7fu;
        if (packed > CHUNK_MASK || tail >= pow3(4) || unpack_chunks(packed) != key) {
            std::cerr << "rankchunk32 23-bit pack mismatch key=" << key
                      << " packed=" << packed << " tail=" << tail << '\n';
            return 2;
        }
        if (packed > max_packed) max_packed = packed;
        if (tail > max_tail) max_tail = tail;
        ++pack_cases;
    }

    uint64_t stripe_cases = 0, lane_checks = 0;
    uint32_t max_blocks = 0;
    for (uint32_t first_off = 0; first_off < BLOCK; ++first_off) {
        const uint32_t first_compact = 1024u + first_off;
        const uint32_t first_block = first_compact >> 5;
        const uint32_t split_lane = BLOCK - first_off;
        for (uint32_t active_lanes = 1; active_lanes <= WARP; ++active_lanes) {
            uint32_t blocks = 1;
            for (uint32_t lane = 0; lane < active_lanes; ++lane) {
                const uint32_t compact = first_compact + lane;
                const uint32_t direct_block = compact >> 5;
                const uint32_t shared_block = first_block
                    + ((split_lane < WARP && lane >= split_lane) ? 1u : 0u);
                if (direct_block != shared_block) {
                    std::cerr << "rankchunk32 block-sharing mismatch first_off=" << first_off
                              << " active=" << active_lanes << " lane=" << lane
                              << " direct=" << direct_block << " shared=" << shared_block << '\n';
                    return 3;
                }
                if (direct_block != first_block) {
                    if (split_lane >= active_lanes || split_lane >= WARP) {
                        std::cerr << "rankchunk32 inactive split source first_off=" << first_off
                                  << " active=" << active_lanes << " split=" << split_lane << '\n';
                        return 4;
                    }
                    blocks = 2;
                }
                ++lane_checks;
            }
            if (blocks > max_blocks) max_blocks = blocks;
            ++stripe_cases;
        }
    }

    if (pack_cases != pow3(K) || stripe_cases != 32u * 32u || max_blocks != 2u)
        return 5;

    std::cout << "gridfp-rankchunk32-warpbase OK"
              << " pack_cases=" << pack_cases
              << " stripe_cases=" << stripe_cases
              << " active_lane_checks=" << lane_checks
              << " chunk_bits=23 prefix_bits=9 block=32"
              << " tail_max=" << max_tail
              << " max_packed=" << max_packed
              << " max_prefix=" << (31u * K)
              << " height_padding_entries=0"
              << " max_block_base_loads_per_warp=" << max_blocks
              << " lane0_source_always_active=1"
              << " split_source_active_if_needed=1"
              << " pack_exact=1 unpack_exact=1 direct_block_exact=1\n";
    return 0;
}
