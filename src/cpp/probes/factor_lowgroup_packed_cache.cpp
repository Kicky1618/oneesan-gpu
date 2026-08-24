#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>

using U64 = std::uint64_t;

static U64 align8(U64 x) { return (x + 7u) & ~U64(7u); }

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const int low = argc > 2 ? std::atoi(argv[2]) : 14;
    const int residues = argc > 3 ? std::atoi(argv[3]) : 1;
    const int high = W - 1 - low;
    if (W < 4 || W > 28 || low < 1 || high < 1 || high >= 16 || residues < 1)
        return 1;

    constexpr U64 FBLOCK_BYTES = 24;
    const U64 masks = U64(1) << high;
    const U64 groups_per_residue = masks * U64(W);
    const U64 v38_base_builds = groups_per_residue * U64(residues);
    const U64 v39_base_builds = masks;
    const U64 packed_bytes = align8(
        U64(64 + 32) * FBLOCK_BYTES
        + 2u * sizeof(int)
        + 2u * sizeof(std::uint32_t)
        + U64(high + 3) * sizeof(std::uint32_t)
        + 2u * U64(high + 2) * sizeof(std::uint32_t));
    const U64 cache_bytes = masks * packed_bytes;

    if (W == 28 && low == 14 && residues == 1) {
        if (masks != 8192ULL
            || groups_per_residue != 229376ULL
            || v38_base_builds != 229376ULL
            || v39_base_builds != 8192ULL
            || packed_bytes != 2504ULL
            || cache_bytes != 20512768ULL) {
            std::cerr << "n=27 packed LOW cache regression\n";
            return 2;
        }
    }

    std::cout << std::fixed << std::setprecision(12)
              << "lowgroup-packed-cache W=" << W
              << " low=" << low << " high=" << high
              << " residues=" << residues << '\n'
              << "masks=" << masks
              << " groups_per_residue=" << groups_per_residue << '\n'
              << "v38_base_builds=" << v38_base_builds
              << " v39_base_builds=" << v39_base_builds
              << " build_reduction="
              << (1.0 - double(v39_base_builds) / double(v38_base_builds)) << '\n'
              << "packed_bytes=" << packed_bytes
              << " host_cache_bytes=" << cache_bytes
              << " host_cache_mib=" << double(cache_bytes) / double(1ULL << 20)
              << '\n';
    return 0;
}
