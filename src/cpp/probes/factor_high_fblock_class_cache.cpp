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
    const U64 main_blocks = U64(3 * (high + 2));
    const U64 block_blocks = U64(high + 2);
    constexpr U64 fblock_bytes = 24;
    const U64 bytes_per_layout = (main_blocks + block_blocks) * fblock_bytes;

    const U64 old_payload = masks * bytes_per_layout;
    const U64 new_payload = classes * bytes_per_layout;
    const U64 old_vector_builds = masks * 2;
    const U64 new_vector_builds = classes * 2;

    if (W == 28 && low == 14) {
        if (classes != 15ULL
            || bytes_per_layout != 1440ULL
            || old_payload != 23592960ULL
            || new_payload != 21600ULL
            || old_vector_builds != 32768ULL
            || new_vector_builds != 30ULL) {
            std::cerr << "n=27 HIGH FBlock class-cache regression mismatch\n";
            return 2;
        }
    }

    std::cout << std::fixed << std::setprecision(9)
              << "high-fblock-class-cache W=" << W
              << " low=" << low << " high=" << high << '\n'
              << "masks=" << masks << " layout_classes=" << classes << '\n'
              << "bytes_per_layout=" << bytes_per_layout << '\n'
              << "old_pinned_bytes=" << old_payload
              << " class_pinned_bytes=" << new_payload
              << " pinned_reduction_pct="
              << 100.0 * (1.0 - double(new_payload) / double(old_payload)) << '\n'
              << "old_vector_builds=" << old_vector_builds
              << " class_vector_builds=" << new_vector_builds
              << " build_reduction_pct="
              << 100.0 * (1.0 - double(new_vector_builds) / double(old_vector_builds))
              << '\n';
    return 0;
}
