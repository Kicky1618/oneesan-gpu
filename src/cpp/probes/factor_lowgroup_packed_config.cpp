#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>

using U64 = std::uint64_t;

static U64 align8(U64 x) { return (x + 7u) & ~U64(7u); }

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const int low = argc > 2 ? std::atoi(argv[2]) : 14;
    const int high = W - 1 - low;
    if (W < 4 || W > 28 || low < 1 || high < 1 || high >= 16) return 1;

    constexpr U64 CODE_BYTES = 8;
    constexpr U64 FBLOCK_BYTES = 24;
    constexpr int MAXW = 28;
    const U64 groups = (U64(1) << high) * U64(W);
    const U64 main_blocks = U64(3 * (high + 2));
    const U64 block_blocks = U64(high + 2);
    const U64 dp_bytes = U64(MAXW + 1) * U64(MAXW + 2) * CODE_BYTES;

    // v0.37 per-group symbol updates:
    //   base: MAIN/BLOCK FBlocks, two nblocks, mask, fix_low,
    //         four fixed/occ words, MAIN/BLOCK DP
    //   compact: Code prefix + u32 low_count
    //   warp-row: u32 prefix + u32 chunks
    const U64 old_base = main_blocks * FBLOCK_BYTES
                       + block_blocks * FBLOCK_BYTES
                       + 2u * sizeof(int)
                       + 6u * sizeof(std::uint32_t)
                       + 2u * dp_bytes;
    const U64 old_compact = U64(high + 3) * CODE_BYTES
                          + U64(high + 2) * sizeof(std::uint32_t);
    const U64 old_warprow = U64(high + 3) * sizeof(std::uint32_t)
                          + U64(high + 2) * sizeof(std::uint32_t);
    const U64 old_bytes_per_group = old_base + old_compact + old_warprow;
    const U64 old_calls_per_group = 16;

    // v0.38 packs full fixed-capacity FBlock arrays and the active u32
    // warp-row plan into one 8-byte-aligned constant object.
    const U64 new_bytes_per_group = align8(
        U64(64 + 32) * FBLOCK_BYTES
        + 2u * sizeof(int)
        + 2u * sizeof(std::uint32_t)
        + U64(high + 3) * sizeof(std::uint32_t)
        + 2u * U64(high + 2) * sizeof(std::uint32_t));
    const U64 new_calls_per_group = 1;

    const U64 old_calls = groups * old_calls_per_group;
    const U64 new_calls = groups * new_calls_per_group;
    const U64 old_bytes = groups * old_bytes_per_group;
    const U64 new_bytes = groups * new_bytes_per_group;

    if (W == 28 && low == 14) {
        if (groups != 229376ULL
            || old_bytes_per_group != 15704ULL
            || new_bytes_per_group != 2504ULL
            || old_calls != 3670016ULL
            || new_calls != 229376ULL
            || old_bytes != 3602120704ULL
            || new_bytes != 574357504ULL) {
            std::cerr << "n=27 packed LOW config regression\n";
            return 2;
        }
    }

    std::cout << std::fixed << std::setprecision(12)
              << "lowgroup-packed-config W=" << W
              << " low=" << low << " high=" << high << '\n'
              << "groups=" << groups
              << " old_calls=" << old_calls
              << " new_calls=" << new_calls
              << " call_reduction="
              << (1.0 - double(new_calls) / double(old_calls)) << '\n'
              << "old_bytes_per_group=" << old_bytes_per_group
              << " new_bytes_per_group=" << new_bytes_per_group << '\n'
              << "old_payload_bytes=" << old_bytes
              << " new_payload_bytes=" << new_bytes
              << " payload_reduction="
              << (1.0 - double(new_bytes) / double(old_bytes)) << '\n';
    return 0;
}
