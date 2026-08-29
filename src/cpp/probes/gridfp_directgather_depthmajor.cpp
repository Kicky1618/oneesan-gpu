#include <cstdint>
#include <iostream>
#include <vector>

int main() {
    constexpr uint32_t depths = 13;
    uint64_t cases = 0, entries = 0;
    for (uint32_t heights = 1; heights <= 30; ++heights) {
        for (uint32_t seed = 1; seed <= 257; seed += 17) {
            std::vector<uint32_t> count(heights), off(heights);
            uint32_t total = 0;
            for (uint32_t h = 0; h < heights; ++h) {
                off[h] = total;
                count[h] = (seed * (h + 3u) * 37u + h * h * 11u) % 4097u;
                total += count[h];
            }
            std::vector<uint64_t> rank_major(size_t(total) * depths);
            std::vector<uint64_t> depth_major(size_t(total) * depths, ~uint64_t(0));
            for (uint32_t h = 0; h < heights; ++h) {
                for (uint32_t r = 0; r < count[h]; ++r) {
                    for (uint32_t d = 0; d < depths; ++d) {
                        const size_t src = (size_t(off[h]) + r) * depths + d;
                        const size_t dst = size_t(off[h]) * depths + size_t(d) * count[h] + r;
                        const uint64_t tag = (uint64_t(h) << 48) | (uint64_t(d) << 32) | r;
                        rank_major[src] = tag;
                        depth_major[dst] = rank_major[src];
                    }
                }
            }
            for (uint32_t h = 0; h < heights; ++h) {
                for (uint32_t d = 0; d < depths; ++d) {
                    const uint32_t depth_off = off[h] * depths + d * count[h];
                    for (uint32_t r = 0; r < count[h]; ++r) {
                        const size_t src = (size_t(off[h]) + r) * depths + d;
                        const size_t dst = size_t(depth_off) + r;
                        if (depth_major[dst] != rank_major[src]) {
                            std::cerr << "depth-major mismatch h=" << h
                                      << " depth=" << d << " rank=" << r << '\n';
                            return 2;
                        }
                        ++entries;
                    }
                }
            }
            ++cases;
        }
    }
    std::cout << "gridfp-directgather-depthmajor-proof OK cases=" << cases
              << " entries=" << entries
              << " depths=13 same_depth_lane_stride_entries=1 exact=1\n";
    return 0;
}
