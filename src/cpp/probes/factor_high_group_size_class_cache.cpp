#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>

using U64 = std::uint64_t;

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const int low = argc > 2 ? std::atoi(argv[2]) : 14;
    const int high = W - 1 - low;
    if (W < 4 || low < 1 || low >= 31 || high < 1) return 1;

    const U64 masks = U64(1) << low;
    const U64 classes = U64(low + 1);
    constexpr U64 sizes_per_slot = 2;
    constexpr U64 code_bytes = 8;
    const U64 old_bytes = masks * sizes_per_slot * code_bytes;
    const U64 new_bytes = classes * sizes_per_slot * code_bytes;
    const U64 old_make_spec = masks * 2;
    const U64 new_make_spec = classes * 2;

    if (W == 28 && low == 14) {
        if (classes != 15ULL
            || old_bytes != 262144ULL
            || new_bytes != 240ULL
            || old_make_spec != 32768ULL
            || new_make_spec != 30ULL) {
            std::cerr << "n=27 HIGH group-size class-cache regression mismatch\n";
            return 2;
        }
    }

    std::cout << std::fixed << std::setprecision(9)
              << "high-group-size-class-cache W=" << W
              << " low=" << low << " high=" << high << '\n'
              << "masks=" << masks << " classes=" << classes << '\n'
              << "old_cache_bytes=" << old_bytes
              << " class_cache_bytes=" << new_bytes
              << " cache_reduction_pct="
              << 100.0 * (1.0 - double(new_bytes) / double(old_bytes)) << '\n'
              << "old_setup_make_spec=" << old_make_spec
              << " class_setup_make_spec=" << new_make_spec
              << " make_spec_reduction_pct="
              << 100.0 * (1.0 - double(new_make_spec) / double(old_make_spec))
              << '\n';
    return 0;
}
