#include <cstdint>
#include <cstdlib>
#include <iostream>

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const int L = argc > 2 ? std::atoi(argv[2]) : 14;
    const int H = argc > 3 ? std::atoi(argv[3]) : W - L - 1;
    if (W < 3 || L < 1 || L >= 31 || H < 1 || L + H + 1 != W) {
        std::cerr << "usage: factor_high_compact_count_release [W L H]\n";
        return 2;
    }

    const std::uint64_t masks = std::uint64_t(1) << L;
    const std::uint64_t cap_stride = std::uint64_t(W / 2 + 1);
    const std::uint64_t low_entries = masks * std::uint64_t(L + 2) * cap_stride;
    const std::uint64_t high_entries = std::uint64_t(H + 2) * cap_stride;
    const std::uint64_t low_bytes = low_entries * sizeof(std::uint16_t);
    const std::uint64_t high_bytes = high_entries * sizeof(std::uint32_t);
    const std::uint64_t released = low_bytes + high_bytes;

    std::cout << "W=" << W
              << " L=" << L
              << " H=" << H
              << " masks=" << masks
              << " low_count_entries=" << low_entries
              << " high_count_entries=" << high_entries
              << " released_bytes=" << released
              << " released_mib=" << double(released) / double(1ULL << 20)
              << '\n';

    if (W == 28 && L == 14 && H == 13 && released != 7865220ULL) {
        std::cerr << "n=27 compact-count release regression got="
                  << released << " expected=7865220\n";
        return 3;
    }
    return 0;
}
