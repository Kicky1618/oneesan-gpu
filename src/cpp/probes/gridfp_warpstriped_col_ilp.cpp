#include <cstdint>
#include <iostream>
#include <vector>

static bool check(uint32_t cols, uint32_t gx, uint32_t ilp) {
    std::vector<uint8_t> seen(cols, 0);
    const uint32_t step = gx * 32u;
    for (uint32_t bx = 0; bx < gx; ++bx) {
        for (uint32_t lane = 0; lane < 32u; ++lane) {
            for (uint32_t base = bx * 32u + lane; base < cols; base += step * ilp) {
                for (uint32_t t = 0; t < ilp; ++t) {
                    const uint32_t lr = base + t * step;
                    if (lr >= cols) continue;
                    if (++seen[lr] != 1u) return false;
                }
            }
        }
    }
    for (uint32_t i = 0; i < cols; ++i) if (seen[i] != 1u) return false;
    return true;
}

int main() {
    uint64_t cases = 0;
    for (uint32_t ilp : {1u, 2u, 4u}) {
        for (uint32_t gx = 1; gx <= 64; ++gx) {
            for (uint32_t cols = 0; cols <= 8192; cols += 7) {
                if (!check(cols, gx, ilp)) {
                    std::cerr << "coverage failure cols=" << cols
                              << " gx=" << gx << " ilp=" << ilp << '\n';
                    return 2;
                }
                ++cases;
            }
        }
    }
    std::cout << "gridfp-warpstriped-col-ilp-proof OK cases=" << cases
              << " ilp=1,2,4 max_cols=8192 max_gx=64 exact_coverage=1\n";
    return 0;
}
