#include <cstdint>
#include <cstdlib>
#include <iostream>

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const int L = argc > 2 ? std::atoi(argv[2]) : 14;
    const int H = argc > 3 ? std::atoi(argv[3]) : W - L - 1;
    if (W < 3 || L < 1 || L >= 31 || H < 1 || L + H + 1 != W) {
        std::cerr << "usage: factor_highclosure_launch_class_cache [W L H]\n";
        return 2;
    }

    const std::uint64_t masks = std::uint64_t(1) << L;
    const std::uint64_t caps = std::uint64_t((W + 1) / 2 + 1);
    const std::uint64_t classes = std::uint64_t(L + 1);
    const std::uint64_t dense_entries = masks * std::uint64_t(H) * caps;
    const std::uint64_t class_entries = classes * std::uint64_t(H) * caps;
    const std::uint64_t dense_bytes = dense_entries * sizeof(std::uint32_t);
    const std::uint64_t class_bytes = class_entries * sizeof(std::uint32_t);

    std::cout << "W=" << W
              << " L=" << L
              << " H=" << H
              << " masks=" << masks
              << " classes=" << classes
              << " dense_entries=" << dense_entries
              << " class_entries=" << class_entries
              << " dense_mib=" << double(dense_bytes) / double(1ULL << 20)
              << " class_mib=" << double(class_bytes) / double(1ULL << 20)
              << " reduction_pct="
              << 100.0 * (1.0 - double(class_entries) / double(dense_entries))
              << '\n';

    if (W == 28 && L == 14 && H == 13) {
        if (dense_entries != 3194880ULL || class_entries != 2925ULL
            || dense_bytes != 12779520ULL || class_bytes != 11700ULL) {
            std::cerr << "n=27 HIGH closure class-cache regression\n";
            return 3;
        }
    }
    return 0;
}
