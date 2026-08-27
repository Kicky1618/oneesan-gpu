#include "../../common/gridfp_closure_pattern10_lut.hpp"

#include <cstdint>
#include <iostream>

using namespace oneesan::gridfp;

int main() {
    auto low = build_closure_pattern10_lut<14>();
    auto high = build_closure_pattern10_lut<13>();
    std::uint64_t payload = std::uint64_t(low.packed.size() + high.packed.size()) * 4ull;
    if (low.packed.size() != 9848u || high.packed.size() != 5645u) {
        std::cerr << "pattern10 LUT production size mismatch low=" << low.packed.size()
                  << " high=" << high.packed.size() << '\n';
        return 2;
    }
    if (payload != 61972ull) {
        std::cerr << "pattern10 LUT payload mismatch bytes=" << payload << '\n';
        return 3;
    }
    std::cout << "gridfp-closure-pattern10-lut-selftest OK low_entries=" << low.packed.size()
              << " high_entries=" << high.packed.size()
              << " payload_bytes=" << payload
              << " payload_kib=" << (double(payload) / 1024.0)
              << '\n';
    return 0;
}
