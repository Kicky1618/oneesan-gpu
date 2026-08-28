#include <array>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {
struct ContextLayoutModel {
    int owner;
    int lo;
    int L;
    std::uint16_t outer_ones;
    std::uint16_t sector_row_base;
    std::uint64_t local_group_base;
};
static_assert(sizeof(ContextLayoutModel) == 24);
constexpr std::array<int,11> BASE{0,24,59,107,170,250,349,469,612,780,975};
}

int main() {
    std::vector<unsigned char> seen(1199, 0);
    std::uint64_t indices = 0;
    int running = 0;
    int max_outer = 0;
    int max_base = 0;
    for (int wi = 0; wi < 11; ++wi) {
        const int W = 8 + 2 * wi;
        const int L = W / 2 + 1;
        const int O = W - L;
        if (BASE[wi] != running) return 2;
        max_outer = std::max(max_outer, O);
        max_base = std::max(max_base, BASE[wi]);
        for (int outer = 0; outer <= O; ++outer) {
            for (int local = 0; local <= L; ++local) {
                const int ix = BASE[wi] + outer * (L + 1) + local;
                if (ix < 0 || ix >= int(seen.size()) || seen[ix]) return 3;
                seen[ix] = 1;
                ++indices;
            }
        }
        running += (O + 1) * (L + 1);
    }
    if (running != 1199 || indices != 1199) return 4;
    for (unsigned char x : seen) if (!x) return 5;
    if (max_outer != 13 || max_base != 975) return 6;
    if (max_outer > 0xffff || max_base > 0xffff) return 7;

    std::cout << "gridfp-runtime-sector-row-cache-proof OK"
              << " W_configs=11"
              << " table_entries=" << indices
              << " context_bytes=" << sizeof(ContextLayoutModel)
              << " max_outer=" << max_outer
              << " max_row_base=" << max_base
              << " uint16_exact=1 contiguous_exact=1 footprint_unchanged=1\n";
    return 0;
}
