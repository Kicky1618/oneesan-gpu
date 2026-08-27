#pragma once

#include "gridfp_closure_pattern10.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace oneesan::gridfp {

template<int K>
struct ClosurePattern10LutHost {
    static_assert(K > 0 && K <= 14, "pattern10 LUT is sized for production half widths <=14");
    std::array<std::uint32_t, K + 2> off{};
    std::vector<std::uint32_t> packed;
    size_t bytes() const { return packed.size() * sizeof(std::uint32_t) + off.size() * sizeof(std::uint32_t); }
};

template<int K>
static ClosurePattern10LutHost<K> build_closure_pattern10_lut() {
    ClosurePattern10LutHost<K> out;
    constexpr int len = K + 1;
    for (int p = 1; p < len; ++p) {
        out.off[size_t(p)] = std::uint32_t(out.packed.size());
        std::uint16_t count = closure_pattern10_count(len, p);
        if (count >= CLOSURE_PATTERN10_NONE) {
            std::cerr << "pattern10 LUT count overflow K=" << K << " p=" << p
                      << " count=" << count << '\n';
            std::exit(580);
        }
        for (std::uint16_t id = 0; id < count; ++id) {
            std::uint16_t lm = 0, rm = 0;
            closure_pattern10_decode(id, len, p, lm, rm);
            out.packed.push_back(std::uint32_t(lm) | (std::uint32_t(rm) << 16));
        }
    }
    out.off[size_t(len)] = std::uint32_t(out.packed.size());
    for (int p = 1; p < len; ++p) {
        std::uint32_t begin = out.off[size_t(p)], end = out.off[size_t(p + 1)];
        std::uint32_t want = closure_pattern10_count(len, p);
        if (end - begin != want) {
            std::cerr << "pattern10 LUT offset mismatch K=" << K << " p=" << p
                      << " got=" << (end - begin) << " want=" << want << '\n';
            std::exit(581);
        }
        for (std::uint16_t id = 0; id < want; ++id) {
            std::uint16_t lm = 0, rm = 0;
            closure_pattern10_decode(id, len, p, lm, rm);
            std::uint32_t z = out.packed[begin + id];
            if (std::uint16_t(z) != lm || std::uint16_t(z >> 16) != rm) {
                std::cerr << "pattern10 LUT decode mismatch K=" << K << " p=" << p
                          << " id=" << id << '\n';
                std::exit(582);
            }
        }
    }
    return out;
}

} // namespace oneesan::gridfp
