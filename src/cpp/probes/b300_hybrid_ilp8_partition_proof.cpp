#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

using Code = std::uint64_t;

static Code blocks_for(Code n, int threads, int lanes) {
    if (!n) return 1;
    const Code cover = Code(threads) * Code(lanes);
    const Code need = (n + cover - 1) / cover;
    return std::min<Code>(65535, std::max<Code>(1, need));
}

static bool inverse_maps(Code i, Code n, int threads, int lanes) {
    if (i >= n) return false;
    const Code blocks = blocks_for(n, threads, lanes);
    const Code grid = blocks * Code(threads);
    const Code tid = i % grid;
    const Code ordinal = i / grid;
    const Code q = ordinal / Code(lanes);
    const Code k = ordinal % Code(lanes);
    const Code base = tid + q * Code(lanes) * grid;
    return tid < grid && k < Code(lanes) && base < n && base + k * grid == i;
}

static bool enumerate_exact(Code n, int threads, int lanes) {
    const Code blocks = blocks_for(n, threads, lanes);
    const Code grid = blocks * Code(threads);
    std::vector<unsigned char> seen(n, 0);
    for (Code tid = 0; tid < grid; ++tid) {
        for (Code base = tid; base < n; base += Code(lanes) * grid) {
            for (int k = 0; k < lanes; ++k) {
                const Code i = base + Code(k) * grid;
                if (i >= n) continue;
                if (seen[i]) return false;
                seen[i] = 1;
            }
        }
    }
    return std::find(seen.begin(), seen.end(), 0) == seen.end();
}

int main() {
    constexpr std::array<int, 6> threads_cases{32, 64, 128, 256, 512, 1024};
    constexpr std::array<Code, 8> thresholds{
        0, 1, 1024, 65536, 262144, 1048576, 4194304, 16777216
    };

    std::uint64_t enumerated = 0;
    std::uint64_t inverse_cases = 0;
    std::uint64_t boundary_cases = 0;

    for (int threads : threads_cases) {
        for (Code threshold : thresholds) {
            std::vector<Code> ns{1, 2, 3, 31, 32, 33, 127, 128, 129, 4095, 4096, 4097};
            if (threshold > 0) ns.push_back(threshold - 1);
            ns.push_back(threshold);
            if (threshold < std::numeric_limits<Code>::max()) ns.push_back(threshold + 1);

            const Code cap4 = Code(65535) * Code(threads) * 4;
            const Code cap8 = Code(65535) * Code(threads) * 8;
            for (Code x : {cap4 - 1, cap4, cap4 + 1, cap8 - 1, cap8, cap8 + 1}) ns.push_back(x);

            for (Code n : ns) {
                if (n == 0) continue;
                const int lanes = n >= threshold ? 8 : 4;
                if ((threshold == 0 && lanes != 8) || (threshold != 0 && n == threshold && lanes != 8)) {
                    std::cerr << "hybrid threshold branch mismatch n=" << n
                              << " threshold=" << threshold << '\n';
                    return 2;
                }
                if (threshold != 0 && n + 1 == threshold && lanes != 4) {
                    std::cerr << "hybrid threshold lower-bound mismatch n=" << n
                              << " threshold=" << threshold << '\n';
                    return 3;
                }
                ++boundary_cases;

                if (n <= 250000) {
                    if (!enumerate_exact(n, threads, lanes)) {
                        std::cerr << "enumeration partition failure n=" << n
                                  << " threads=" << threads << " lanes=" << lanes << '\n';
                        return 4;
                    }
                    ++enumerated;
                }

                const Code last = n - 1;
                std::array<Code, 12> samples{
                    0,
                    std::min<Code>(1, last),
                    std::min<Code>(Code(threads) - 1, last),
                    std::min<Code>(Code(threads), last),
                    last / 7,
                    last / 5,
                    last / 3,
                    last / 2,
                    last > 2 ? last - 2 : last,
                    last > 1 ? last - 1 : last,
                    last,
                    std::min<Code>(Code(65535) * Code(threads), last),
                };
                for (Code i : samples) {
                    if (!inverse_maps(i, n, threads, lanes)) {
                        std::cerr << "inverse partition failure n=" << n
                                  << " i=" << i << " threads=" << threads
                                  << " lanes=" << lanes << " threshold=" << threshold << '\n';
                        return 5;
                    }
                    ++inverse_cases;
                }
            }
        }
    }

    std::cout << "b300-hybrid-ilp8-partition-proof OK"
              << " threshold_rule=n_ge_threshold_uses_ilp8"
              << " ilp4_destinations=4 ilp8_destinations=8"
              << " block_cap=65535 exact_partition=1"
              << " enumerated_cases=" << enumerated
              << " inverse_cases=" << inverse_cases
              << " boundary_cases=" << boundary_cases << '\n';
    return 0;
}
