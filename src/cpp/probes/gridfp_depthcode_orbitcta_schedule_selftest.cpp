#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

static bool check_case(int total, int cols, int threads, int gx, int gy) {
    if (threads <= 0 || threads > 1024 || (threads & 31) || gx <= 0 || gy <= 0)
        return false;
    const int nwarps = threads / 32;
    std::vector<uint8_t> warp(size_t(total) * cols, 0);
    std::vector<uint8_t> orbitcta(size_t(total) * cols, 0);
    auto hit = [&](std::vector<uint8_t>& a, int k, int lr) {
        if (k < 0 || k >= total || lr < 0 || lr >= cols) std::abort();
        if (++a[size_t(k) * cols + lr] != 1) return false;
        return true;
    };

    // Current warpstriped HIGH schedule.
    for (int by = 0; by < gy; ++by)
        for (int bx = 0; bx < gx; ++bx)
            for (int w = 0; w < nwarps; ++w)
                for (int lane = 0; lane < 32; ++lane)
                    for (int k = by * nwarps + w; k < total; k += gy * nwarps)
                        for (int lr = bx * 32 + lane; lr < cols; lr += gx * 32)
                            if (!hit(warp, k, lr)) return false;

    // New orbit-CTA schedule.  grid.x is exactly one; all CTA threads cover the
    // complete column row while grid.y distributes orbits.
    const int orbit_gy = gy * nwarps; // same nominal orbit lanes as warpstriped.
    for (int by = 0; by < orbit_gy; ++by)
        for (int t = 0; t < threads; ++t)
            for (int k = by; k < total; k += orbit_gy)
                for (int lr = t; lr < cols; lr += threads)
                    if (!hit(orbitcta, k, lr)) return false;

    for (size_t i = 0; i < warp.size(); ++i)
        if (warp[i] != 1 || orbitcta[i] != 1) return false;
    return warp == orbitcta;
}

int main() {
    uint64_t cases = 0, cells = 0;
    for (int threads : {32, 64, 128, 256, 512})
        for (int gx : {1, 2, 4, 8, 16, 32})
            for (int gy : {1, 2, 4, 8, 16})
                for (int total : {1, 2, 7, 8, 9, 31, 32, 33, 67, 129})
                    for (int cols : {1, 31, 32, 33, 127, 128, 129, 1025}) {
                        if (!check_case(total, cols, threads, gx, gy)) {
                            std::cerr << "orbitcta schedule mismatch threads=" << threads
                                      << " gx=" << gx << " gy=" << gy
                                      << " total=" << total << " cols=" << cols << '\n';
                            return 1;
                        }
                        ++cases;
                        cells += uint64_t(total) * uint64_t(cols);
                    }
    std::cout << "gridfp-depthcode-orbitcta-schedule-selftest OK cases=" << cases
              << " cells=" << cells
              << " exact_orbit_column_cover=1"
              << " warpstriped_context_builds_per_orbit=gx"
              << " orbitcta_context_builds_per_orbit=1\n";
}
