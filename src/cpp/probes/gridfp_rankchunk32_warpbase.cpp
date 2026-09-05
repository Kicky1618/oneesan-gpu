#include <array>
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
    static_assert(CHUNK_BITS + PREFIX_BITS == 32u);

    constexpr std::array<uint32_t, 5> PREFIX_PROBES{0u, 1u, 255u, 256u, 511u};
    uint64_t pack_cases = 0, prefix_isolation_cases = 0;
    uint32_t max_packed = 0, max_tail = 0;
    for (uint32_t key = 0; key < pow3(K); ++key) {
        const uint32_t packed = pack_chunks(key);
        const uint32_t tail = (packed >> 16) & 0x7fu;
        if (packed > CHUNK_MASK || tail >= pow3(4) || unpack_chunks(packed) != key) {
            std::cerr << "rankchunk32 23-bit pack mismatch key=" << key
                      << " packed=" << packed << " tail=" << tail << '\n';
            return 2;
        }
        for (uint32_t prefix : PREFIX_PROBES) {
            const uint32_t meta = packed | (prefix << CHUNK_BITS);
            const uint32_t c0 = meta & 0xffu;
            const uint32_t c1 = (meta >> 8) & 0xffu;
            const uint32_t c2 = (meta >> 16) & 0x7fu;
            const uint32_t got_prefix = meta >> CHUNK_BITS;
            const uint32_t reconstructed = c0 * pow3(9) + c1 * pow3(4) + c2;
            if (reconstructed != key || got_prefix != prefix || c2 != key % pow3(4)) {
                std::cerr << "rankchunk32 prefix/chunk alias key=" << key
                          << " prefix=" << prefix << " meta=" << meta
                          << " c2=" << c2 << " got_prefix=" << got_prefix << '\n';
                return 3;
            }
            ++prefix_isolation_cases;
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
        const uint32_t split_lane = BLOCK - first_off; // [1,32]
        for (uint32_t active_lanes = 1; active_lanes <= WARP; ++active_lanes) {
            uint32_t blocks = 1;
            for (uint32_t lane = 0; lane < active_lanes; ++lane) {
                const uint32_t compact = first_compact + lane;
                const uint32_t direct_block = compact >> 5;
                const bool second = split_lane < WARP && lane >= split_lane;
                const uint32_t shared_block = first_block + uint32_t(second);
                const uint32_t source_lane = second ? split_lane : 0u;
                if (direct_block != shared_block) {
                    std::cerr << "rankchunk32 block-sharing mismatch first_off=" << first_off
                              << " active=" << active_lanes << " lane=" << lane
                              << " direct=" << direct_block << " shared=" << shared_block << '\n';
                    return 4;
                }
                if (source_lane >= active_lanes) {
                    std::cerr << "rankchunk32 one-shuffle source inactive first_off=" << first_off
                              << " active=" << active_lanes << " lane=" << lane
                              << " source=" << source_lane << " split=" << split_lane << '\n';
                    return 5;
                }
                const uint32_t source_block = (first_compact + source_lane) >> 5;
                if (source_block != direct_block) {
                    std::cerr << "rankchunk32 one-shuffle source block mismatch first_off="
                              << first_off << " active=" << active_lanes << " lane=" << lane
                              << " source=" << source_lane << " source_block=" << source_block
                              << " direct_block=" << direct_block << '\n';
                    return 6;
                }
                if (second) blocks = 2;
                ++lane_checks;
            }
            if (blocks > max_blocks) max_blocks = blocks;
            ++stripe_cases;
        }
    }

    if (pack_cases != pow3(K) ||
        prefix_isolation_cases != pack_cases * PREFIX_PROBES.size() ||
        stripe_cases != 32u * 32u || max_blocks != 2u)
        return 7;

    std::cout << "gridfp-rankchunk32-warpbase OK"
              << " pack_cases=" << pack_cases
              << " prefix_isolation_cases=" << prefix_isolation_cases
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
              << " pack_exact=1 unpack_exact=1 prefix_isolation_exact=1"
              << " direct_block_exact=1 one_shuffle_source_exact=1\n";
    return 0;
}
