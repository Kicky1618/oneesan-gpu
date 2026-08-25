#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const int low = argc > 2 ? std::atoi(argv[2]) : 14;
    const int high = W - 1 - low;
    if (W < 4 || low < 1 || high < 1 || high >= 31) return 1;

    const std::uint64_t masks = std::uint64_t(1) << high;
    // Current ABI: FBlock=24 bytes, 64 MAIN + 32 BLOCKED + 16 scalar bytes.
    const std::uint64_t static_bytes = 96ULL * 24ULL + 16ULL;
    const std::uint64_t dynamic_bytes = std::uint64_t(high + 3) * 4ULL
        + std::uint64_t(high + 2) * 8ULL
        + std::uint64_t(low) * 65ULL * 12ULL
        + std::uint64_t(high + 2) * 4ULL;
    const std::uint64_t raw_config = static_bytes + dynamic_bytes;
    const std::uint64_t config_bytes = (raw_config + 7ULL) & ~7ULL;
    const std::uint64_t old_cache = masks * config_bytes;
    const std::uint64_t new_cache = masks * static_bytes;

    if (W == 28 && low == 14) {
        if (static_bytes != 2320ULL || config_bytes != 13488ULL
            || old_cache != 110493696ULL || new_cache != 19005440ULL) {
            std::cerr << "n=27 static LOW cache regression\n";
            return 2;
        }
    }

    std::cout << std::fixed << std::setprecision(6)
              << "lowgroup-static-cache W=" << W << " low=" << low
              << " high=" << high << '\n'
              << "static_bytes_per_mask=" << static_bytes
              << " full_config_bytes_per_mask=" << config_bytes << '\n'
              << "old_cache_mib=" << double(old_cache) / double(1ULL << 20)
              << " new_cache_mib=" << double(new_cache) / double(1ULL << 20)
              << " reduction=" << 1.0 - double(new_cache) / double(old_cache)
              << '\n';
    return 0;
}
